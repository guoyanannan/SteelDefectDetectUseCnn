
"""
Run inference on steel images
"""
import argparse
import os.path
import glob
import torch
import time
from multiprocessing import Process, Queue
from detect_yolov5.utils.data_process import get_input_tensor,read_images,get_steel_edge
from detect_yolov5.utils.general import check_img_size, print_args
from detect_yolov5.steel_detect_yolo import YOLOInit
from detect_yolov5.utils.normaloperation import LOGS


#(img_arr_rgb, cam_resolution, imgsz, stride, device, pt)
def data_tensor_infer(q,model_obj,cam_resolution,img_resize,stride,device,auto,conf_thres,iou_thres,classes,agnostic_nms,max_det,debug,logger):
    model_obj.to(device)
    while 1:
        if not q.empty():
            img_infos = q.get()
            img_arr_rgb, img_path, left_eg, right_eg = tuple(img_infos.values())
            img_ori_one_shape, img_ori_one_scale_tensor, img_cut_two_shape, img_cut_two_scale_tensor, img_cut_bs_shape, img_cut_bs_scale_tensor \
                = get_input_tensor(img_arr_rgb, cam_resolution, img_resize, stride, device, auto)
            model_obj.pre_process_detect(img_path,
                                     './result_img_1',
                                     left_eg,
                                     right_eg,
                                     img_arr_rgb,
                                     img_cut_bs_scale_tensor,
                                     img_cut_bs_shape,
                                     img_ori_one_scale_tensor,
                                     img_ori_one_shape,
                                     img_cut_two_scale_tensor,
                                     img_cut_two_shape,
                                     conf_thres,
                                     iou_thres,
                                     classes,
                                     agnostic_nms,
                                     max_det,
                                     cam_resolution,
                                     debug=debug
                                     )

        else:
            logger.info('暂时没有数据了,等待！！！！！！！！！！！！！！！！！')
            time.sleep(1)

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        log_path='./detect_logs',
        dirs=('data/images',),
        imgsz=(640, 640),
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        bin_thres=0.35,
        edge_shift=0,
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        cam_resolution='4k',  # camera resolution
        schema=1,  # 0：有算法测试程序，1：没有算法测试程序
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        is_test=False,
        debug=False,  # debug mode
        ):

    source = dirs
    my_log = LOGS(log_path)
    model = YOLOInit(weights, gpu_cpu=device, half=half, log_op=my_log,augment=augment,visualize=visualize ,dnn=dnn)
    stride, names, pt, device = model.stride, model.names, model.pt,model.device
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print(source,stride, names, pt, device,imgsz)
    read_queue = Queue(100)
    queue_list = [Queue(100)]
    read_pro = Process(target=read_images, args=(source,read_queue,schema,my_log))
    select_edge_pro = Process(target=get_steel_edge, args=(read_queue,queue_list,schema,edge_shift,bin_thres, cam_resolution, my_log))
    read_pro.start()
    select_edge_pro.start()
    for i in range(len(queue_list)):
        q_get_info = queue_list[i]
        run_pro = Process(target=data_tensor_infer,
                          args=(q_get_info,
                                model,
                                cam_resolution,
                                imgsz,
                                stride,
                                device,
                                pt,
                                conf_thres,
                                iou_thres,
                                classes,
                                agnostic_nms,
                                max_det,
                                debug,
                                my_log
                                )
                          )
        run_pro.start()

    # while 1:
    #     for i in range(len(queue_list)):
    #         if not queue_list[i].empty():
    #             img_infos = queue_list[i].get()
    #             img_arr_rgb,img_path,left_eg,right_eg = tuple(img_infos.values())
    #             img_ori_one_shape, img_ori_one_scale_tensor, img_cut_two_shape, img_cut_two_scale_tensor, img_cut_bs_shape, img_cut_bs_scale_tensor = get_input_tensor(img_arr_rgb, cam_resolution, imgsz, stride, device, pt)
    #             model.pre_process_detect(img_path,
    #                                      './result_img_1',
    #                                      left_eg,
    #                                      right_eg,
    #                                      img_arr_rgb,
    #                                      img_cut_bs_scale_tensor,
    #                                      img_cut_bs_shape,
    #                                      img_ori_one_scale_tensor,
    #                                      img_ori_one_shape,
    #                                      img_cut_two_scale_tensor,
    #                                      img_cut_two_shape,
    #                                      conf_thres,
    #                                      iou_thres,
    #                                      classes,
    #                                      agnostic_nms,
    #                                      max_det,
    #                                      cam_resolution,
    #                                      debug=True
    #                                      )
    #             # cv2.line(img_arr_rgb, (le, 0), (le, img_arr_rgb.shape[0] - 1), (255, 0, 0), 3)
    #             # cv2.line(img_arr_rgb, (re, 0), (re, img_arr_rgb.shape[0] - 1), (255, 0, 0), 3)
    #             # cv2.namedWindow('edge', 0)
    #             # cv2.imshow('edge', img_arr_rgb)
    #             # cv2.waitKey(0)
    #
    #         else:
    #             continue

    # print('开始时间：',time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    # count_num = 0
    # time_count = 0
    # model_infer_time_count = 0
    # time_read = 0
    # process_time = time_sync()

    # img_num_every = len(images)//2
    # data_1 = images[:img_num_every]
    # data_2 = images[img_num_every:]

    #
    # pro_1 = Process(target=img_read,args=(source,read_queue))
    # # pro_3 = Process(target=img_read,args=(source_1,read_queue))
    # pro_2 = Process(target=get_lr,args=(read_queue,))
    # # pro_3 = Process(target=read_get,args=(source,read_queue))
    # # pro_4 = Process(target=get_lr_test, args=(read_queue,))
    # pro_1.start()
    # pro_2.start()
    # # pro_3.start()
    # # pro_4.start()
    # print('启动时间:', time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        # for file_path in images:
        #     t_1 = time_sync()
        #     img_arr_rgb = cv2.cvtColor(cv2.imread(file_path,0),cv2.COLOR_GRAY2RGB)
        #     t_1_1 = time_sync()
        #     time_read += (t_1_1-t_1)
        #
        #     left_eg,right_eg,img_ori_one_shape, img_ori_one_scale_tensor,img_cut_two_shape, img_cut_two_scale_tensor, img_cut_bs_shape, img_cut_bs_scale_tensor = get_input_tensor(img_arr_rgb,cam_resolution,0,imgsz,stride,device,pt)
        #     print(f'left_eg:{left_eg}')
        #     print(f'right_eg:{right_eg}')
        #     print(f'img_ori_one_shape:{img_ori_one_shape}')
        #     print(f'img_ori_one_scale_tensor:{img_ori_one_scale_tensor.size()}')
        #     print(f'img_cut_two_shape：{img_cut_two_shape}')
        #     print(f'img_cut_two_scale_tensor:{img_cut_two_scale_tensor.size()}')
        #     print(f'img_cut_bs_shape:{img_cut_bs_shape}')
        #     print(f'img_cut_bs_scale_tensor:{img_cut_bs_scale_tensor.size()}')
        #     # cv2.line(img_arr_rgb, (left_eg, 0), (left_eg, img_arr_rgb.shape[0] - 1), (255, 0, 0), 3)
        #     # cv2.line(img_arr_rgb, (right_eg, 0), (right_eg, img_arr_rgb.shape[0] - 1), (255, 0, 0), 3)
        #     # cv2.namedWindow('edge', 0)
        #     # cv2.imshow('edge', img_arr_rgb)
        #     # cv2.waitKey(0)
        #     t_2 = time_sync()
        #     time_count += (t_2-t_1)
        #     t_3 = time_sync()
        #     draw_path = './result_img_1'
        #     # 原始图像路径,五张图缩放后拼成batch的tensor,原图缩放后拼成的batchtensor，五张图无缩放拼接成的tensor，原图无缩放tensor，
        #     # path, img_five_s, img_ori_s, img_five,img_ori,img_draw,
        #     model.pre_process_detect(file_path,
        #                              draw_path,
        #                              left_eg,
        #                              right_eg,
        #                              img_arr_rgb,
        #                              img_cut_bs_scale_tensor,
        #                              img_cut_bs_shape,
        #                              img_ori_one_scale_tensor,
        #                              img_ori_one_shape,
        #                              img_cut_two_scale_tensor,
        #                              img_cut_two_shape,
        #                              conf_thres,
        #                              iou_thres,
        #                              classes,
        #                              agnostic_nms,
        #                              max_det,
        #                              cam_resolution,
        #                              debug=False
        #                              )
        #     t_4 = time_sync()
        #     model_infer_time_count += (t_4-t_3)
        #     count_num += 1
        #     print(f'数据读取时间:{time_read/count_num}s,数据处理平均耗时：{(time_count / count_num)-(time_read/count_num)},'
        #           f'模型推理时间平均耗时：{model_infer_time_count/count_num}s,'
        #           f'共耗时：{(time_count / count_num)+(model_infer_time_count/count_num)}s')


    '''
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    total_time = 0
    img_num = 0
    #test
    draw_path = './result_img'
    #path, img_splits,img_ori_to_bs, img0_splits, img0,self.cap, s
    #_,_
    for path, img_five_s, img_ori_s, img_five,img_ori,img_draw,vid_cap, s in dataset:
        t1 = time_sync()
        model.pre_process_detect(path,
                                 draw_path,
                                 0,
                                 4095,
                                 img_draw,
                                 img_five_s,
                                 img_five.shape,
                                 img_ori_s,
                                 img_ori.shape,
                                 conf_thres,
                                 iou_thres,
                                 classes,
                                 agnostic_nms,
                                 max_det,
                                 cam_resolution,
                                 debug=True
                                 )
        total_time += time_sync() - t1
        img_num += 1
        print(f'每张图片切分5张+原图++nms总时间为:{(time_sync() - t1) * 1000}ms,平均耗时:{(total_time / img_num) * 1000}ms,'
              f'process fps:{1 / (total_time / img_num)},fps:{6 / (total_time / img_num)}')

        # if debug:
        #     print('五张图缩放后拼成batchtensor:',img_five_s.shape)
        #     print('五张图无缩放拼接成batchtensor:',img_five.shape)
        #     print('原图缩放后拼成的batchtensor:', img_ori_s.shape)
        #     print('原图无缩放tensor:', img_ori.shape)
        # img_get_shape = (img_five_s,img_ori_s)
        # img_get_shape_ori = (img_five,img_ori)
        # t1 = time_sync()
        # pred_five = model(img_five_s, augment=augment, visualize=visualize)#(5,10647,6) tensor
        # pred_ori = model(img_ori_s, augment=augment, visualize=visualize) #(1,3276,6)   tensor
        # 
        # # NMS
        # # on (n,6) tensor per image [xyxy, conf, cls]
        # pred_five = non_max_suppression(pred_five, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)#[tensor([]),...,tensor((n,6)),(),()]  len =5
        # pred_ori = non_max_suppression(pred_ori, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)#[tensor([]),...,tensor((n,6)),(),()]   len = 1
        # pred_two = (pred_five,pred_ori)
        # 
        # # list of element tensor(1,6) for box
        # one_result_for_img = []
        # five_result_for_img = []
        # # 首先进行坐标转换
        # for i in range(len(pred_two)):
        #     preds = pred_two[i]
        #     # 查看每个批次多少张图片
        #     preds_len = len(preds)
        #     for j in range(preds_len):
        #         pred_every_img = preds[j] #tensor (n,6)-> box number
        #         if len(pred_every_img):
        #             # 将坐标转换至图像原始尺寸 return *xyxy, conf, cls
        #             pred_every_img[:, :4] = scale_coords(img_get_shape[i].shape[-2:], pred_every_img[:, :4],img_get_shape_ori[i].shape[-2:]).round()
        #             # 将切分图坐标转换至原始图像
        #             if preds_len > 1:
        #                 pred_every_img[:,0], pred_every_img[:,2] = pred_every_img[:,0]+j*768, pred_every_img[:,2]+j*768
        #                 for index in range(len(pred_every_img)):
        #                     five_result_for_img.append(pred_every_img[index])
        #             elif preds_len == 1:
        #                 for index in range(len(pred_every_img)):
        #                     one_result_for_img.append(pred_every_img[index])
        # # margeBox
        # if len(five_result_for_img):
        #     five_result_for_img = torch.vstack(five_result_for_img)
        #     five_result_for_img = marge_box(five_result_for_img, names, iou_thresh=0.05,debug=False)
        # if len(one_result_for_img):
        #     one_result_for_img = torch.vstack(one_result_for_img)
        # 
        # if len(five_result_for_img) and not len(one_result_for_img):
        #     nms_ = five_result_for_img
        # elif not len(five_result_for_img) and len(one_result_for_img):
        #     nms_ = one_result_for_img
        # elif not len(five_result_for_img) and not len(one_result_for_img):
        #     nms_ = []
        # else:
        #     nms_ = torch.vstack([one_result_for_img,five_result_for_img])
        # # nms
        # if len(nms_):
        #     nms_ = xyxy_img_nms(nms_, names, iou_thresh=0.05)
        # 
        # total_time += time_sync() - t1
        # img_num += 1
        # print(f'每张图片切分5张+原图++nms总时间为:{(time_sync() - t1)*1000}ms,平均耗时:{(total_time / img_num)*1000}ms,'
        #       f'process fps:{1/(total_time / img_num)},fps:{6/(total_time / img_num)}')
        # 
        # for i,box in enumerate(nms_):
        #     label = f'detect:{box[-2]:.2f}'
        #     color = (0,0,255)
        #     txt_color = (255,255,255)
        #     lw = 3
        #     p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        #     cv2.rectangle(img_draw, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        #     if label:
        #         tf = max(lw - 1, 1)  # font thickness
        #         w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        #         outside = p1[1] - h - 3 >= 0  # label fits outside box
        #         p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        #         cv2.rectangle(img_draw, p1, p2, color, -1, cv2.LINE_AA)  # filled
        #         cv2.putText(img_draw, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
        #                     thickness=tf, lineType=cv2.LINE_AA)
        # if debug:
        #     cv2.namedWindow('1',0)
        #     cv2.imshow('1',img_draw)
        #     cv2.waitKey(0)
        # file_name = os.path.basename(path)
        # path_save = os.path.join('result',file_name)
        # if not os.path.exists('result/'):
        #     os.makedirs('result/')
        # cv2.imwrite(path_save,img_draw)
    '''

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/bests.pt', help='model path(s)')
    parser.add_argument('--log_path', type=str, default='./detect_logs', help='Directory where log files reside')
    parser.add_argument('--dirs', type=tuple, default=('data/images',), help='image directory')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--bin-thres', type=float, default=0.35, help='image binaryzation threshold')
    parser.add_argument('--edge-shift', type=float, default=0, help='left and right edge shift')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cam_resolution',type=str, default='4k', help='camera resolution,use 4k 0r 8k')
    parser.add_argument('--schema', type=int, default=1, help='Whether there is an algorithm test program')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--is_test', action='store_true', help='use test for inference')
    parser.add_argument('--debug', action='store_true',default=True, help='use debug mode')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args('detect', opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    print(vars(opt))
    main(opt)
