# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os.path
import torch
import time
from detect_yolov5.utils.datasets import LoadImages
from detect_yolov5.utils.general import (check_img_size, cv2, non_max_suppression, print_args, scale_coords)
ROOT = os.path.dirname(os.path.realpath(__file__))
print('ROOT:',ROOT)

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        cam_resolution='',  # camera resolution
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        debug=False,  # debug mode
        ):
    source = str(source)
    from detect_yolov5.steel_detect_yolo import YOLOInit
    from detect_yolov5.utils.normaloperation import LOGS,marge_box,xyxy_img_nms
    my_log = LOGS('./detect_logs')
    model = YOLOInit(weights, gpu_cpu=device, half=half, log_op=my_log,dnn=dnn)
    stride, names, pt,device = model.stride, model.names, model.pt,model.device
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    total_time = 0
    img_num = 0
    #path, img_splits,img_ori_to_bs, img0_splits, img0,self.cap, s
    #原始图像路径,五张图缩放后拼成batch的tensor,原图缩放后拼成的batchtensor，五张图无缩放拼接成的tensor，原图无缩放tensor，_,_
    for path, img_five_s, img_ori_s, img_five,img_ori,img_draw,vid_cap, s in dataset:
        if debug:
            print('五张图缩放后拼成batchtensor:',img_five_s.shape)
            print('五张图无缩放拼接成batchtensor:',img_five.shape)
            print('原图缩放后拼成的batchtensor:', img_ori_s.shape)
            print('原图无缩放tensor:', img_ori.shape)
        img_get_shape = (img_five_s,img_ori_s)
        img_get_shape_ori = (img_five,img_ori)
        t1 = time_sync()
        pred_five = model(img_five_s, augment=augment, visualize=visualize)#(5,10647,6) tensor
        pred_ori = model(img_ori_s, augment=augment, visualize=visualize) #(1,3276,6)   tensor

        # NMS
        # on (n,6) tensor per image [xyxy, conf, cls]
        pred_five = non_max_suppression(pred_five, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)#[tensor([]),...,tensor((n,6)),(),()]  len =5
        pred_ori = non_max_suppression(pred_ori, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)#[tensor([]),...,tensor((n,6)),(),()]   len = 1
        pred_two = (pred_five,pred_ori)

        # list of element tensor(1,6) for box
        one_result_for_img = []
        five_result_for_img = []
        # 首先进行坐标转换
        for i in range(len(pred_two)):
            preds = pred_two[i]
            # 查看每个批次多少张图片
            preds_len = len(preds)
            for j in range(preds_len):
                pred_every_img = preds[j] #tensor (n,6)-> box number
                if len(pred_every_img):
                    # 将坐标转换至图像原始尺寸 return *xyxy, conf, cls
                    pred_every_img[:, :4] = scale_coords(img_get_shape[i].shape[-2:], pred_every_img[:, :4],img_get_shape_ori[i].shape[-2:]).round()
                    # 将切分图坐标转换至原始图像
                    if preds_len > 1:
                        pred_every_img[:,0], pred_every_img[:,2] = pred_every_img[:,0]+j*768, pred_every_img[:,2]+j*768
                        for index in range(len(pred_every_img)):
                            five_result_for_img.append(pred_every_img[index])
                    elif preds_len == 1:
                        for index in range(len(pred_every_img)):
                            one_result_for_img.append(pred_every_img[index])
        # margeBox
        if len(five_result_for_img):
            five_result_for_img = torch.vstack(five_result_for_img)
            five_result_for_img = marge_box(five_result_for_img, names, iou_thresh=0.05,debug=False)
        if len(one_result_for_img):
            one_result_for_img = torch.vstack(one_result_for_img)

        if len(five_result_for_img) and not len(one_result_for_img):
            nms_ = five_result_for_img
        elif not len(five_result_for_img) and len(one_result_for_img):
            nms_ = one_result_for_img
        elif not len(five_result_for_img) and not len(one_result_for_img):
            nms_ = []
        else:
            nms_ = torch.vstack([one_result_for_img,five_result_for_img])
        # nms
        if len(nms_):
            nms_ = xyxy_img_nms(nms_, names, iou_thresh=0.05)

        total_time += time_sync() - t1
        img_num += 1
        print(f'每张图片切分5张+原图++nms总时间为:{(time_sync() - t1)*1000}ms,平均耗时:{(total_time / img_num)*1000}ms,'
              f'process fps:{1/(total_time / img_num)},fps:{6/(total_time / img_num)}')

        for i,box in enumerate(nms_):
            label = f'detect:{box[-2]:.2f}'
            color = (0,0,255)
            txt_color = (255,255,255)
            lw = 3
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(img_draw, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(img_draw, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img_draw, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)
        if debug:
            cv2.namedWindow('1',0)
            cv2.imshow('1',img_draw)
            cv2.waitKey(0)
        file_name = os.path.basename(path)
        path_save = os.path.join('result',file_name)
        if not os.path.exists('result/'):
            os.makedirs('result/')
        cv2.imwrite(path_save,img_draw)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/bests.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cam_resolution', default='4k', help='camera resolution,use 4k 0r 8k')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', default=True, help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--debug', action='store_true', help='use debug mode')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args('detect', opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
