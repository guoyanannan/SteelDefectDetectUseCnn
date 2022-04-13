
"""
Run inference on steel images
"""
import argparse
import os.path
import cv2
import torch
import time
from multiprocessing import Process, Queue
from detect_yolov5.utils.data_process import get_input_tensor,read_images,get_steel_edge
from detect_yolov5.utils.general import check_img_size, print_args
from detect_yolov5.steel_detect_yolo import YOLOInit
from detect_yolov5.utils.normaloperation import LOGS


#(img_arr_rgb, cam_resolution, imgsz, stride, device, pt)
def data_tensor_infer(q,result_roi_q,model_obj,cam_resolution,img_resize,stride,device,auto,conf_thres,iou_thres,classes,agnostic_nms,max_det,debug,logger):

    model_obj.to(device)
    currt_num = 1
    total_time = 0
    while 1:
        if not q.empty():
            start_time = time.time()
            img_infos = q.get()
            img_arr_rgb, img_path, left_eg, right_eg = tuple(img_infos.values())
            img_ori_one_shape, img_ori_one_scale_tensor, img_cut_two_shape, img_cut_two_scale_tensor, img_cut_bs_shape, img_cut_bs_scale_tensor \
                = get_input_tensor(img_arr_rgb, cam_resolution, img_resize, stride, device, auto)
            model_obj.pre_process_detect(img_path,
                                     result_roi_q,
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
            end_time = time.time()
            currt_num +=1
            total_time += end_time-start_time
            print(f'process-{os.getpid()}累计处理了{currt_num}张图片,平均耗时{total_time/currt_num}s')
        else:
            logger.info(f'process-{os.getpid()}-结束时间：{time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())},累计处理了{currt_num}张图片,平均耗时{total_time/currt_num}s')
            #print(f'{os.getpid()}-结束时间：', time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
            time.sleep(1)


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        log_path='./detect_logs',
        dirs=('data/images',),
        rois_dir='./result_roi',
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
        debug=False,  # debug mode
        ):

    source = dirs
    logger_ = LOGS(log_path)
    model = YOLOInit(weights, gpu_cpu=device, half=half, log_path=log_path,augment_=augment,visualize_=visualize, dnn=dnn)
    stride, names, pt, device = model.stride, model.names, model.pt,model.device
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print(source,stride, names, pt, device,imgsz)
    read_queue = Queue(300)
    roi_q = Queue(1000)
    queue_list = [Queue(100),Queue(100),Queue(100)]
    read_pro = Process(target=read_images, args=(source,read_queue,schema,log_path))
    select_edge_pro = Process(target=get_steel_edge, args=(read_queue,queue_list,schema,edge_shift,bin_thres, cam_resolution, log_path))
    read_pro.start()
    select_edge_pro.start()

    for i in range(len(queue_list)):
        print(f'进程{i}')
        q_get_info = queue_list[i]
        run_pro = Process(target=data_tensor_infer,
                          args=(q_get_info,
                                roi_q,
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
                                LOGS(log_path)
                                )
                          )
        run_pro.start()
    print('开始时间：', time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    while 1:
        if not roi_q.empty():
            roi_infos = roi_q.get()
            img_roi,img_name = tuple(roi_infos.values())
            if not os.path.exists(rois_dir):
                os.makedirs(rois_dir)
            path_save = os.path.join(rois_dir, img_name)
            cv2.imwrite(path_save, img_roi)
        else:
            print(f'存储小图的队列暂时没有数据了,等待......................')
            time.sleep(0.5)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/bests.pt', help='model path(s)')
    parser.add_argument('--log_path', type=str, default='./detect_logs', help='Directory where log files reside')
    parser.add_argument('--rois_dir', type=str, default='./result_roi', help='Directory where log files reside')
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
