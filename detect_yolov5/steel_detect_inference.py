
"""
Run inference on steel images
"""
import argparse
import os.path
import sys
import signal
import cv2
import torch
import time
import yaml
from threading import Thread
from multiprocessing import Process, Queue,freeze_support
from detect_yolov5.utils.data_process import data_tensor_infer,read_images,get_steel_edge
from detect_yolov5.utils.general import check_img_size, print_args
from detect_yolov5.steel_detect_yolo import YOLOInit
from detect_yolov5.utils.normaloperation import re_print,delete_temp

# ROOT = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(ROOT)

@torch.no_grad()
def run(
        weights='yolov5s.pt',  # model.pt path(s)
        log_path='./detect_logs',
        dirs=('data/images',),
        rois_dir='./result_roi',
        imgsz=(640, 640),
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        bin_thres=0.35,
        edge_shift=0,
        max_det=1000,  # maximum detections per image
        pro_num=3,  # process number
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

    pid_list = []
    source = dirs
    # logger_ = LOGS(log_path)
    try:
        model = YOLOInit(weights, gpu_cpu=device, half=half, log_path=log_path, augment_=augment, visualize_=visualize,
                         dnn=dnn)
    except Exception as E:
        re_print(E)
        raise E
    stride, names, pt, device = model.stride, model.names, model.pt,model.device
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    some_info = f'图像路径:{source} 显卡索引{device} 图像缩放尺寸{imgsz}'
    model.log_op.info(some_info)
    # read image q
    read_queue = Queue(500)
    # roi sets q
    roi_q = Queue()
    # q required for Detection
    queue_list=[]
    for i in range(pro_num):
        queue_list.append(Queue(100))
    #
    read_pro = Process(target=read_images, args=(source, read_queue, schema, log_path))
    read_pro.start()
    model.log_op.info(f'process-{read_pro.pid} starting success')
    select_edge_pro = Process(target=get_steel_edge,
                              args=(read_queue, queue_list, schema, edge_shift, bin_thres, cam_resolution, log_path))
    select_edge_pro.start()
    model.log_op.info(f'process-{select_edge_pro.pid} starting success')
    # pid number list
    pid_list += [read_pro,select_edge_pro]

    for i in range(len(queue_list)):
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
                                half,
                                debug,
                                log_path
                                )
                          )
        run_pro.start()
        pid_list.append(run_pro)
        model.log_op.info(f'process-{run_pro.pid} starting success')

    # delete temp files
    del_thread = Thread(target=delete_temp, args=(r'C:\Users\{}\AppData\Local\Temp'.format(os.getlogin()),))
    del_thread.start()
    while 1:
        try:
            # if a process fails, kill them all
            for pip in pid_list:
                if not pip.is_alive():
                    for pip_kill in pid_list:
                        pip_kill.terminate()
                    os.kill(os.getpid(), signal.SIGINT)

            if not roi_q.empty():
                roi_infos = roi_q.get()
                img_roi,img_name = tuple(roi_infos.values())
                if not os.path.exists(rois_dir):
                    os.makedirs(rois_dir)
                path_save = os.path.join(rois_dir, img_name)
                cv2.imwrite(path_save, img_roi)
            else:
                re_print(f'缺陷数据队列中暂时没有了,等待')
                time.sleep(0.1)
        except Exception as E:
            model.log_op.info(E)
            for pip in pid_list:
                if pip.is_alive():
                    for pip_kill in pid_list:
                        pip_kill.terminate()
                    os.kill(os.getpid(), signal.SIGINT)


def parse_opt():
    print(os.path.abspath(__file__))
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cg = yaml.load(f.read(), Loader=yaml.FullLoader)
    detect_config = cg['detect']
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=detect_config['filePath']['weights'], help='model path(s)')
    parser.add_argument('--log_path', type=str, default=detect_config['filePath']['logs'], help='Directory where log files reside')
    parser.add_argument('--rois_dir', type=str, default=detect_config['filePath']['rois'], help='Directory where roi images reside')
    parser.add_argument('--dirs', type=tuple, default=tuple(detect_config['filePath']['srcs']), help='image directory')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[detect_config['infoParameter']['img_size']], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=detect_config['infoParameter']['conf_thres'], help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=detect_config['infoParameter']['iou_thres'], help='NMS IoU threshold')
    parser.add_argument('--bin-thres', type=float, default=detect_config['infoParameter']['bin_thres'], help='image binaryzation threshold')
    parser.add_argument('--edge-shift', type=float, default=detect_config['infoParameter']['edge_shift'], help='left and right edge shift')
    parser.add_argument('--max-det', type=int, default=detect_config['infoParameter']['max_det'], help='maximum detections per image')
    parser.add_argument('--pro_num', type=int, default=detect_config['infoParameter']['pro_num'],help='the number of processes detection')
    parser.add_argument('--device', default=detect_config['infoParameter']['device'], help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cam_resolution',type=str, default=detect_config['infoParameter']['cam_res'], help='camera resolution,use 4k 0r 8k')
    parser.add_argument('--schema', type=int, default=detect_config['infoParameter']['schema'], help='Whether there is an algorithm test program')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', default=False, help='augmented inference')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize features')
    parser.add_argument('--half', action='store_true', default=detect_config['infoParameter']['half'], help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', default=detect_config['infoParameter']['dnn'], help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--debug', action='store_true', default=detect_config['infoParameter']['debug'], help='use debug mode')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args('detect', opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    freeze_support()
    opt = parse_opt()
    print(vars(opt))
    main(opt)
