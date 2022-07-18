import math
import os
import json
import torch
import cv2
import sys
import time
import pandas as pd
import torch.nn as nn
import numpy as np
from threading import Thread
from detect_yolov5.models.experimental import attempt_load
from detect_yolov5.utils.normaloperation import select_device,LOGS,marge_box,xyxy_img_nms,thread_save_rois
from detect_yolov5.utils.general import non_max_suppression,scale_coords
ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.append(ROOT)


class YOLOInit(nn.Module):

    def __init__(self, weights, gpu_cpu, half, log_path,augment_,visualize_ ,dnn=False):
        super().__init__()
        self.weights = weights
        self.log_op = LOGS(log_path)
        self.fp16 = half
        self.dnn = dnn
        self.device = select_device(gpu_cpu, self.log_op)
        w = str(self.weights[0] if isinstance(self.weights, list) else self.weights)
        pt, jit, onnx = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.fp16 &= (pt or jit or onnx) and self.device.type != 'cpu'  # FP16
        if pt:  # PyTorch
            self.log_op.info(f'Loading {w} for PyTorch inference...')
            model = attempt_load(weights if isinstance(weights, list) else w)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if self.fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            self.log_op.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            model.half() if self.fp16 else model.float()
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif self.dnn:  # ONNX OpenCV DNN
            self.log_op.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            self.log_op.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        self.__dict__.update(locals())

    def forward(self, im, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=self.augment_, visualize=self.visualize_)
            return y if val else y[0]
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            if self.device.type != 'cpu':
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 320, 320)):
        if any((self.pt, self.jit, self.onnx, self.dnn)):  # warmup types
            if self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                for _ in range(2 if self.jit else 1):  #
                    a = self.forward(im)  # warmup
                    # print(a.shape)

    def pre_process_detect(self,
                           im_path,  # 用于存图等的逻辑撰写
                           roi_dir_path,
                           result_roi_q,
                           left_edge,
                           right_edge,
                           img_draw,  # 画坐标框用于可视化
                           im_bs_scale,  # 数组形状(bs,c,h,w)
                           im_bs_ori_shape,  # 形状可选(bs,c,h,w)or(c,h,w)or(h,w)
                           im_one_scale,  # 数组形状(1,c,h,w)
                           im_one_orin_shape,  # 形状可选(1,c,h,w)or(c,h,w)or(h,w)
                           im_two_scale,  # 数组形状(2,c,h,w)
                           im_two_orin_shape,  # 形状可选(2,c,h,w)or(c,h,w)or(h,w)
                           conf_thres,
                           iou_thres,
                           classes,
                           agnostic_nms,
                           max_det,
                           cam_resolution,
                           schema,
                           debug=False,  #debug mode
                           ):

        # if debug:
        #     print('多张图缩放后拼成batchtensor:', im_bs_scale.shape)
        #     print('多张图无缩放的尺寸:', im_bs_ori_shape)
        #     print('原图缩放后拼成的batchtensor:', im_one_scale.shape)
        #     print('原图无缩放的尺寸:', im_one_orin_shape)
        if cam_resolution.lower() == '4k':
            split_stride = 768
        elif cam_resolution.lower() == '8k':
            split_stride = 896
            im_one_scale = im_two_scale
            im_one_orin_shape = im_two_orin_shape
        # input node data size
        img_get_shape = (im_bs_scale.shape, im_one_scale.shape)
        # scale previous data size
        img_get_shape_ori = (im_bs_ori_shape, im_one_orin_shape)
        # (5,10647,6) tensor 4k
        # (9,10647,6) tensor 8k
        pred_bs = self.model(im_bs_scale,augment=self.augment_, visualize=self.visualize_)[0]

        # (1,3276,6) tensor 4k
        # (2,3276,6) tensor 8k
        pred_ori = self.model(im_one_scale, augment=self.augment_, visualize=self.visualize_)[0]

        # NMS
        # on (n,6) tensor per image [xyxy, conf, cls]
        # [tensor([]),...,tensor((n,6)),(),()]  len =5 4k
        # [tensor([]),...,tensor((n,6)),(),()]  len =9 8k
        pred_bs = non_max_suppression(pred_bs, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # [tensor([]),...,tensor((n,6)),(),()]   len = 1 4k
        # [tensor([]),...,tensor((n,6)),(),()]   len = 2 8k
        pred_ori = non_max_suppression(pred_ori, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        #
        pred_two = (pred_bs, pred_ori)

        # list of element tensor(1,6) for box
        one_result_for_img = []
        bs_result_for_img = []
        # 首先进行坐标转换
        for i in range(len(pred_two)):
            preds = pred_two[i]
            # 查看每个批次多少张图片
            preds_len = len(preds)
            for j in range(preds_len):
                pred_every_img = preds[j]  # tensor (n,6)-> box number
                if len(pred_every_img):
                    # 将坐标转换至图像原始尺寸 return *xyxy, conf, cls
                    pred_every_img[:, :4] = scale_coords(img_get_shape[i][-2:], pred_every_img[:, :4],
                                                         img_get_shape_ori[i][-2:]).round()
                    # 将切分图坐标转换至原始图像
                    if preds_len >= 4:
                        pred_every_img[:, 0], pred_every_img[:, 2] = pred_every_img[:, 0] + j * split_stride, pred_every_img[:,2] + j * split_stride
                        for index in range(len(pred_every_img)):
                            bs_result_for_img.append(pred_every_img[index])
                    elif preds_len==2:
                        split_stride_ = img_get_shape_ori[i][-1]
                        pred_every_img[:, 0], pred_every_img[:, 2] = pred_every_img[:, 0] + j * split_stride_, pred_every_img[:,2] + j * split_stride_
                        for index in range(len(pred_every_img)):
                            one_result_for_img.append(pred_every_img[index])
                    elif preds_len == 1:
                        for index in range(len(pred_every_img)):
                            one_result_for_img.append(pred_every_img[index])

        # margeBox
        if len(bs_result_for_img):
            bs_result_for_img = torch.vstack(bs_result_for_img)
            bs_result_for_img = marge_box(bs_result_for_img, self.names, iou_thresh=0.005, debug=False)
        if len(one_result_for_img):
            one_result_for_img = torch.vstack(one_result_for_img)

        if len(bs_result_for_img) and not len(one_result_for_img):
            nms_ = bs_result_for_img
        elif not len(bs_result_for_img) and len(one_result_for_img):
            nms_ = one_result_for_img
        elif not len(bs_result_for_img) and not len(one_result_for_img):
            nms_ = []
        else:
            nms_ = torch.vstack([one_result_for_img, bs_result_for_img])

        # nms
        if len(nms_):
            nms_ = xyxy_img_nms(nms_, self.names, iou_thresh=0.05)

        if debug:
            img_draw_ = img_draw.copy()
        # t1 = time.time()
        # img_split = cv2.cvtColor(img_draw, cv2.COLOR_RGB2GRAY)
        # t2 = time.time()
        img_split = img_draw[:,:,0]
        # t3 = time.time()
        # print(f'=======================1时间{t2-t1}s,2时间{t3-t2}s=========================')
        h_img_ori,w_img_ori = im_one_orin_shape[-2:]
        if cam_resolution.lower() == '8k':
            h_img_ori, w_img_ori = im_one_orin_shape[-2],im_one_orin_shape[-1]*2
        file_name, file_mat = os.path.splitext(os.path.basename(im_path))
        roi_list_ = []

        for i, box in enumerate(nms_):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label = self.names[int(box[-1])]
            score = str(int(box[-2]*100))
            # 越边界
            if x2 <= left_edge or x1 >= right_edge:
                continue
            # 扩展ROI
            x1_cut,y1_cut,x2_cut,y2_cut = x1-25, y1-25, x2+25, y2+25
            if x1_cut < 0:
                x1_cut = 1
            if y1_cut < 0:
                y1_cut = 1
            if x2_cut > w_img_ori:
                x2_cut = w_img_ori-1
            if y2_cut > h_img_ori:
                y2_cut = h_img_ori-1

            # box在 ROI的 位置
            # cut roi from image
            x1_roi,y1_roi = abs(x1-x1_cut),abs(y1-y1_cut)
            x2_roi,y2_roi = x1_roi+(x2-x1),y1_roi+(y2-y1)
            img_roi = img_split[y1_cut:y2_cut,x1_cut:x2_cut]
            if not schema:
                if abs(x2-x1)*abs(y2-y1) >= 1024 * 1024:
                    img_roi = cv2.resize(img_roi, (img_roi.shape[1]//4, img_roi.shape[0]//4))
                    x1_roi, y1_roi, x2_roi, y2_roi = x1_roi//4, y1_roi//4, x2_roi//4, y2_roi//4
                elif abs(x2-x1) >= 512:
                    img_roi = cv2.resize(img_roi, (img_roi.shape[1]//3, img_roi.shape[0]))
                    x1_roi,x2_roi = x1_roi//3, x2_roi//3
                if debug:
                    img_roi_draw = cv2.cvtColor(img_roi, cv2.COLOR_GRAY2RGB)
                    cv2.rectangle(img_roi_draw, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
            #
            # # 画ROI中的框
            # im_roi_draw = im_roi.copy()
            # im_roi_draw = cv2.cvtColor(im_roi_draw,cv2.COLOR_GRAY2RGB)
            # cv2.rectangle(im_roi_draw, (x1_roi,y1_roi), (x2_roi,y2_roi), (255,0,0), thickness=3, lineType=cv2.LINE_AA)
            
            # img_roi = img_split[y1_cut:y2_cut, x1_cut:x2_cut]
            
            # if abs(x2-x1) > 20:
            #     img_roi = cv2.resize(img_roi,(img_roi.shape[1]//4,img_roi.shape[0]))
            #     x1,x2 = x1 // 4 ,x2 //4

            if debug:
                if not schema:
                    roi_file_name = '_'.join([file_name, str(i),
                                              str(x1_roi), str(x2_roi), str(y1_roi), str(y2_roi),
                                              str(x1), str(x2), str(y1), str(y2), label, score
                                              ]) + str(file_mat)
                else:
                    roi_file_name = '_'.join([file_name, str(i),
                                              str(x1), str(x2), str(y1), str(y2), label, score
                                              ]) + str(file_mat)
            else:
                flag_str,steel_no,camera_no,img_index,left_pos,top_pos,fx,fy,_ = file_name.split('_')
                left_in_steel, top_in_steel = int(float(left_pos) + x1 * float(fx)), int(float(top_pos) + y1 * float(fy))
                right_in_steel, bot_in_steel = int(float(left_pos) + x2 * float(fx)), int(float(top_pos) + y2 * float(fy))
                if not schema:
                    roi_file_name = '_'.join([str(flag_str), str(steel_no), str(camera_no), str(img_index),
                                              str(left_edge), str(right_edge),
                                              str(x1_roi), str(x2_roi), str(y1_roi), str(y2_roi),
                                              str(x1), str(x2), str(y1), str(y2),
                                              str(left_in_steel), str(right_in_steel), str(top_in_steel), str(bot_in_steel),
                                              label, score,
                                              'H'
                                              ]) + str(file_mat)
                else:
                    roi_file_name = '_'.join([str(flag_str), str(steel_no),str(camera_no),str(img_index),
                                              str(left_edge),str(right_edge),
                                              str(x1), str(x2), str(y1), str(y2),
                                              str(left_in_steel), str(right_in_steel), str(top_in_steel), str(bot_in_steel),
                                              label, score,
                                              'H'
                                              ]) + str(file_mat)
            if debug and not schema:
                img_roi_info = {'data': img_roi_draw, 'name': roi_file_name}
            else:
                img_roi_info = {'data': img_roi, 'name': roi_file_name}
            # result_roi_q.put(img_roi_info)
            roi_list_.append(img_roi_info)
            
            if debug:
                # 画图中的框
                p1, p2 = (x1,y1), (x2,y2)
                label = f'{self.names[int(box[-1])]}:{box[-2]:.2f}'
                color,txt_color, lw= (0, 0, 255), (255, 255, 255), 3
                cv2.rectangle(img_draw_, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                # 写框的信息
                tf = max(lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(img_draw_, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img_draw_, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
                # 边界
                cv2.line(img_draw_, (left_edge, 0), (left_edge, img_draw_.shape[0] - 1), (255, 0, 0), 3)
                cv2.line(img_draw_, (right_edge, 0), (right_edge, img_draw_.shape[0] - 1), (255, 0, 0), 3)

        if debug:
            im_draw_path = './debug_result/images_4k/'
            if cam_resolution == '8k':
                im_draw_path = './debug_result/images_8k/'
            if not os.path.exists(im_draw_path):
                os.makedirs(im_draw_path)
            file_name = os.path.basename(im_path)
            path_save = os.path.join(im_draw_path,file_name)
            cv2.imwrite(path_save,img_draw_)
        # start = time.time()
        # roi_list_bs_no = math.ceil(len(roi_list_)/2)
        # th_roi_list = []
        # for i in range(2):
        #     th_save = Thread(target=thread_save_rois,args=(roi_list_[i*roi_list_bs_no:(i+1)*roi_list_bs_no],roi_dir_path,))
        #     th_save.start()
        #     th_roi_list.append(th_save)
        # for thd in th_roi_list:
        #     thd.join()
        # for thd_ in th_roi_list:
        #     del thd_
        # del th_roi_list

        thread_save_rois(roi_list_,roi_dir_path)
        # end = time.time()
        # print(f'process-{os.getpid()} kkkkkkkkkkkkkkk 存储{len(roi_list_)}张图片耗时{end-start}s kkkkkkkkkkkkkkkkkkk')


    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        suffixes = list(YOLOInit.export_formats().Suffix)  # export suffixes
        YOLOInit.check_suffix(p, suffixes)  # checks
        p = os.path.basename(p)  # eliminate trailing separators
        print(suffixes)
        pt, jit, onnx = (s in p for s in suffixes)
        return pt, jit, onnx

    @staticmethod
    def export_formats():
        # YOLOv5 export formats
        x = [['PyTorch', '-', '.pt', True],
             ['TorchScript', 'torchscript', '.torchscript', True],
             ['ONNX', 'onnx', '.onnx', True]]
        return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'GPU'])

    @staticmethod
    def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
        # Check file(s) for acceptable suffix
        if file and suffix:
            if isinstance(suffix, str):
                suffix = [suffix]
            for f in file if isinstance(file, (list, tuple)) else [file]:
                s = os.path.splitext(f)[-1] # file suffix
                if len(s):
                    assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


if __name__ == '__main__':
    my_log = LOGS('./detect_logs')
    YOLOInit(0, '0', 0, my_log)