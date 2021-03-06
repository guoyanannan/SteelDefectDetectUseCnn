import os
import time
import signal
import platform
import logging
import tensorflow as tf
from PIL import Image



def re_print(info):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {info}....................')


def print_args(name, opt):
    # Print argparser arguments
    logging.info(f'{name}: ' + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))


def select_device(device,batch_size=0, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'classifier tensorflow {tf.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[int(device)], True)
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[int(device)],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)]
        )
        s += f'GPU:{device} '
    cuda = not cpu
    if cuda:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device))  # i.e. 0,1,6,7
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle).total /1024 /1024
        info = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        s += f'({info},{int(meminfo)}MiB)'

    else:
        s += 'CPU'

    if not newline:
        s = s.rstrip()
    s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s  # emoji-safe
    return s


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for filename in os.listdir(path):
            filepath = os.path.join(path,filename)
            try:
                os.remove(filepath)
            except Exception as E:
                re_print(f'??????{filename}?????????{E}')


def select_no_img(file_path_list,dirname):
    if file_path_list:
        dir_path_dst = file_path_list.copy()
        img_mat = ['bmp', 'jpg', 'BMP', 'JPG', 'png']
        for filepath in file_path_list:
            filename = os.path.basename(filepath)
            if filename.split('.')[-1] not in img_mat:
                num = 1
                while True:
                    try:
                        os.remove(filepath)
                        dir_path_dst.remove(filepath)
                        re_print(f'??????{filename}????????????????????????,????????????')
                        break
                    except Exception as E:
                        re_print(E)
                        num += 1
                        time.sleep(0.1)
                        if num > 10:
                            break
        re_print(f'??????????????????{dirname}??? {len(file_path_list)} ???????????????????????????????????????????????????:{len(dir_path_dst)}')
        return dir_path_dst


def delete_batch_file(batch_path):
    for file_path in batch_path:
        # ??????????????????10??????????????????????????????????????????????????????
        num = 1
        while 1:
            try:
                os.remove(file_path)
                break
            except Exception as E:
                re_print(E)
                num += 1
                time.sleep(0.1)
                if num > 10:
                    break


def delete_dir(dir_path):
    files_path = [os.path.join(dir_path, fileName) for fileName in os.listdir(dir_path)]
    if files_path:
        delete_batch_file(files_path)


def parse_data(info_iter,save_intercls,save_dir_path,curr_defect_cam_num,class_ignore,score_ignore,curr_schema,debug):
    defect_infos = [[] for i in range(len(curr_defect_cam_num))]
    for infos in info_iter:
        img_path, img_roi, class_name, internal_no, external_no, score = infos
        if curr_schema['flag']==0:
            _, steel_no, cam_no, img_no, left_edge, right_edge, \
                roi_x1, roi_x2, roi_y1, roi_y2, \
                img_x1, img_x2, img_y1, img_y2, \
                steel_x1, steel_x2, steel_y1, steel_y2, _, _, _ = os.path.basename(img_path).split('_')
        else:
            _, steel_no, cam_no, img_no, left_edge, right_edge, \
            img_x1, img_x2, img_y1, img_y2, \
            steel_x1, steel_x2, steel_y1, steel_y2, _, _, _ = os.path.basename(img_path).split('_')
        fx = abs(int(steel_x2)-int(steel_x1))/abs(int(img_x2)-int(img_x1))
        fy = abs(int(steel_y2)-int(steel_y1))/abs(int(img_y2)-int(img_y1))
        # ???????????????????????????
        if int(internal_no) in eval(save_intercls):
            img_roi_pil = Image.fromarray(img_roi)
            savedir_ = os.path.join(save_dir_path, steel_no)
            savedir_intercls = os.path.join(savedir_, class_name)
            if not os.path.exists(savedir_intercls):
                while 1:
                    try:
                        os.makedirs(savedir_intercls)
                        break
                    except Exception as E:
                        re_print(E)
                        time.sleep(0.1)
            img_name = f'{class_name}_score({int(score)})_{os.path.basename(img_path)}'
            new_path = os.path.join(savedir_intercls, img_name)
            img_roi_pil.save(new_path)
        # ??????????????????????????????
        if int(external_no) != class_ignore and float(score) > float(score_ignore):
            defect_id = curr_defect_cam_num[cam_no] + 1
            leftToEdge = int(abs(int(img_x1)-int(left_edge)) * fx)
            rightToEdge = int(abs(int(right_edge)-int(img_x2)) * fx)
            area = (int(steel_x2)-int(steel_x1))*(int(steel_y2)-int(steel_y1))
            grade, cycle = 0, 0
            if curr_schema['flag'] == 0:
                try:
                    # ?????????
                    if curr_schema['flag'] == 0 and not debug:
                        img_roi_pil = Image.fromarray(img_roi)
                        dir_path = curr_schema[f'camera{cam_no}']
                        save_dir_ = os.path.join(dir_path,steel_no)
                        if not os.path.exists(save_dir_):
                            while 1:
                                try:
                                    os.makedirs(save_dir_)
                                    break
                                except Exception as E:
                                    re_print(f'????????????{save_dir_}????????????:{E}')
                                    time.sleep(0.1)
                        img_path = os.path.join(save_dir_, f'{defect_id}.bmp')
                        img_roi_pil.save(img_path)
                    # ??????????????????
                    curr_roi_info = (int(defect_id),
                                     int(cam_no),
                                     int(steel_no),
                                     int(img_no),
                                     int(external_no),
                                     int(roi_x1),
                                     int(roi_x2),
                                     int(roi_y1),
                                     int(roi_y2),
                                     int(img_x1),
                                     int(img_x2),
                                     int(img_y1),
                                     int(img_y2),
                                     int(steel_x1),
                                     int(steel_x2),
                                     int(steel_y1),
                                     int(steel_y2),
                                     int(grade),
                                     int(area),
                                     int(leftToEdge),
                                     int(rightToEdge),
                                     int(cycle)
                                     )
                    defect_infos[int(cam_no)-1].append(curr_roi_info)
                except Exception as E:
                    re_print(f'?????????????????????????????????:{E}')
                    os.kill(os.getpid(), signal.SIGINT)
                else:
                    curr_defect_cam_num[cam_no] = defect_id
            else:
                grade = int(score)
                img_data = None
                try:
                    # ??????????????????
                    curr_roi_info = (int(0),
                                     int(steel_no),
                                     int(cam_no),
                                     int(img_no),
                                     int(external_no),
                                     int(grade),
                                     int(img_x1),
                                     int(img_x2),
                                     int(img_y1),
                                     int(img_y2),
                                     int(steel_x1),
                                     int(steel_x2),
                                     int(steel_y1),
                                     int(steel_y2),
                                     int(area),
                                     int(cycle),
                                     img_data,
                                     )
                    defect_infos[int(cam_no) - 1].append(curr_roi_info)
                except Exception as E:
                    re_print(f'????????????????????????:{E}')
                    os.kill(os.getpid(), signal.SIGINT)
                else:
                    curr_defect_cam_num[cam_no] = defect_id
    defect_infos = [tuple(ls) for ls in defect_infos]
    return curr_defect_cam_num,tuple(defect_infos)










