import os
import os.path
import cv2
import glob
import torch
import time
import json
import yaml
import gevent
import math
import numpy as np
from threading import Thread
from .normaloperation import LOGS, re_print
from .db_mysql import DbMysqlOp

IMG_FORMATS = 'bmp', 'jpeg', 'jpg',  'png'

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)  寻找缩放比例最小的边，如果由比例比例都大于 1，那么最小缩放比例便为 1 ，按照最小缩放比例进行缩放
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding 如果要填充的+w,+h 正好是64的倍数，片不在进行填充；否则填充余数值
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def edge_select(img_src, radio, shift):
    h,w = img_src.shape
    select_rows = [int(h*1/32),int(h*8/32),int(h*16/32),int(h*24/32),int(h*31/32)]
    select_rows = np.array(select_rows,dtype=np.uint)
    rows_ = img_src[select_rows,:].mean(axis=0)
    index = []
    # left
    flag = True
    for i in range(w):
        if rows_[i] > 0 and flag:
            index.append(i)
            flag = False
        if rows_[i] == 0 and not flag:
            index.append(i)
            flag = True
        if i==w-1 and len(index) % 2 ==1:
            index.append(i)
        #图片全是背景时
        if i==w-1 and len(index)==0:
            index += [0,w-1]

    #计算最大间隔即为边界
    index_arr = np.array(index,dtype=np.uint).reshape((-1,2))
    img_sub = index_arr[:,1]-index_arr[:,0]
    max_edge = (index_arr[np.argmax(img_sub)]*radio).tolist()
    left_edge,right_edge = max_edge[0]+shift,max_edge[1]-shift
    if left_edge < 0:
        left_edge=0
    if right_edge > radio * w:
        right_edge = radio * w
    return left_edge,right_edge


def select_edge(img_arr,shift,bin_threshold=0.35,cam_resolution='4k'):
    img = cv2.medianBlur(img_arr, 5)
    if cam_resolution.lower() == '4k'.lower():
        h_scale,w_scale = 4, 4
    if cam_resolution.lower() == '8k'.lower():
        h_scale,w_scale = 4,8
    img = cv2.resize(img, (img.shape[1] // w_scale, img.shape[0] // h_scale))
    img = cv2.blur(img, (7, 7))
    img_left = img[:, :img.shape[1] // 2]
    img_right = img[:, img.shape[1] // 2:]
    max_mean = max(cv2.mean(img_left)[0], cv2.mean(img_right)[0])
    _, img_th = cv2.threshold(img, max_mean * bin_threshold, 255, 0)
    return edge_select(img_th,w_scale,shift)


def numpy_to_torch_tensor(img_arr,img_size,stride,auto,device,half):
    img_arr = letterbox(img_arr, img_size, stride, auto=auto)[0]
    img_arr = img_arr.transpose((2, 0, 1))
    img_arr = np.ascontiguousarray(img_arr)
    img_arr = torch.from_numpy(img_arr).to(device)
    img_arr = img_arr.half() if half else img_arr.float()
    img_arr /= 255
    if len(img_arr.shape) == 3:
        img_bs_scale = img_arr[None]
        return img_bs_scale
    else:
        raise Exception()


def remove_file(file_path,logger):
    is_remv = False
    for i in range(3):
        try:
            os.remove(file_path)
            is_remv = True
            break
        except:
            time.sleep(0.01)
            continue
    if not is_remv:
        logger.info(f'{file_path} deletion failed and the next loop will be entered')


def get_input_tensor(img_arr,  # RGB
               cam_resolution,  # 相机分辨率
               img_size,  # 期望尺寸 [h,w]
               stride,    # 网络最终下采样倍数
               device,    # str cuda:0 or 1 or 2,  or str cpu
               auto,
               fp16,
               bin_threshold=0.35,  #
               ):

    img_ori = img_arr  # RGB
    img_split = img_arr
    if cam_resolution.lower() == '4k'.lower():
        # 原图 - 1张 - 原始形状 - bchw
        img_ori_one_shape = (1, img_ori.shape[-1], img_ori.shape[0], img_ori.shape[1])
        # 原图 - 1张 - 缩放形态 - bchw
        img_ori_one_scale_tensor = numpy_to_torch_tensor(img_ori, img_size, stride, auto, device, fp16)

        # 对于4k图像来说，不会切分两张，在此是为了和8k的统一加入队列的标准
        # 原图 - 2张 - 原始形状 - bchw
        img_cut_two_shape = img_ori_one_shape
        # 原图 - 2张 - 缩放形态 - bchw
        img_cut_two_scale_tensor = img_ori_one_scale_tensor

        # 切图
        win_stride = 768
        flag_temp = True
        img_cut_list = []
        img_cut_ori = []
        for r in range(0, (img_split.shape[0] - img_split.shape[0]) + 1, win_stride):  # H方向进行切分
            for c in range(0, (img_split.shape[1] - 1024) + 1, win_stride):  # W方向进行切分
                tmp = img_split[r: r + img_split.shape[0], c: c + 1024]
                # 切图 - 原始形状 - bchw
                if flag_temp:
                    img_cut_bs_shape_ = (1, tmp.shape[-1], tmp.shape[0], tmp.shape[1])
                    img_cut_ori.append(img_cut_bs_shape_)
                    flag_temp = False
                # 切图 - 缩放形态 - bchw
                img_cut_one_scale_tensor = numpy_to_torch_tensor(tmp, img_size, stride, auto, device, fp16)
                img_cut_list.append(img_cut_one_scale_tensor)

        img_cut_bs_scale_tensor = torch.cat(img_cut_list, dim=0)
        img_cut_bs_shape = img_cut_ori[0]

    if cam_resolution.lower() == '8k'.lower():
        # 原图 - 1张 - 原始形状 - bchw
        img_ori_one_shape = (1, img_ori.shape[-1], img_ori.shape[0], img_ori.shape[1])
        # 原图 - 1张 - 缩放形态 - bchw
        img_ori_one_scale_tensor = numpy_to_torch_tensor(img_ori, img_size, stride, auto, device, fp16)

        # 原图 - 2张 - 原始形状 - bchw
        img_cut_two_shape = (2, img_ori.shape[-1], img_ori.shape[0], img_ori.shape[1] // 2)
        # 原图 - 2张 - 缩放形态 - bchw
        img_two_list = []
        img_cut_two_scale_tensor_1 = numpy_to_torch_tensor(img_ori[:, :img_ori.shape[1] // 2], img_size, stride, auto,
                                                           device, fp16)
        img_cut_two_scale_tensor_2 = numpy_to_torch_tensor(img_ori[:, img_ori.shape[1] // 2:], img_size, stride, auto,
                                                           device, fp16)
        img_two_list += [img_cut_two_scale_tensor_1, img_cut_two_scale_tensor_2]
        img_cut_two_scale_tensor = torch.cat(img_two_list, dim=0)
        # 切图
        win_stride = 896
        flag_temp = True
        img_cut_list = []
        img_cut_ori = []
        for r in range(0, (img_split.shape[0] - img_split.shape[0]) + 1, win_stride):  # H方向进行切分
            for c in range(0, (img_split.shape[1] - 1024) + 1, win_stride):  # W方向进行切分
                tmp = img_split[r: r + img_split.shape[0], c: c + 1024]
                # 切图 - 原始形状 - bchw
                if flag_temp:
                    img_cut_bs_shape_ = (1, tmp.shape[-1], tmp.shape[0], tmp.shape[1])
                    img_cut_ori.append(img_cut_bs_shape_)
                    flag_temp = False
                # 切图 - 缩放形态 - bchw
                img_cut_one_scale_tensor = numpy_to_torch_tensor(tmp, img_size, stride, auto, device, fp16)
                img_cut_list.append(img_cut_one_scale_tensor)

        img_cut_bs_scale_tensor = torch.cat(img_cut_list)
        img_cut_bs_shape = img_cut_ori[0]

    return \
           img_ori_one_shape, img_ori_one_scale_tensor,\
           img_cut_two_shape, img_cut_two_scale_tensor, \
           img_cut_bs_shape, img_cut_bs_scale_tensor

def event_reads(path,shift, bin_th, cam_res, q_read, curr_img_num):
    # 获取图片数组
    _, file_mat = os.path.splitext(os.path.basename(path))
    # read images
    while 1:
        try:
            img_arr = cv2.imread(path, 0)
            img_h, img_w = img_arr.shape
            l_e, r_e = select_edge(img_arr, shift, bin_th, cam_res)
            break
        except Exception as E:
            re_print(E)
            continue
    # read infos of images
    while 1:
        try:
            # 获取图像相关信息
            with open(path, 'ab+') as fp:
                fp.seek(-292, 1)
                res_ = fp.read(292)
                res_info = eval(res_.split(b'\x00')[0].decode())
            steel_no_bmp, img_index, cam_no_bmp, steel_start, steel_end, steel_left, steel_right = tuple(
                res_info.values())
            break
        except Exception as E:
            re_print(E)
            continue
    # 解析数据
    fx = (float(steel_right) - float(steel_left)) / img_w
    fy = (float(steel_end) - float(steel_start)) / img_h
    img_name = '_'.join(['SCILRTB', str(steel_no_bmp), str(cam_no_bmp),
                         str(img_index), str(steel_left), str(steel_start),
                         str(fx), str(fy), 'H']) + str(file_mat)
    dict_info = {'img_gray': img_arr, 'img_path': img_name, 'left_e': l_e, 'right_e': r_e}
    q_list_index = curr_img_num % len(q_read)
    q_read[q_list_index].put(dict_info)

def thread_reads(paths,shift, bin_th, cam_res, q_read,name):
    for path in paths:
        # 获取图片数组
        _, file_mat = os.path.splitext(os.path.basename(path))
        # read images
        while 1:
            try:
                img_arr = cv2.imread(path, 0)
                img_h, img_w = img_arr.shape
                l_e, r_e = select_edge(img_arr, shift, bin_th, cam_res)
                break
            except Exception as E:
                re_print(E)
                time.sleep(0.001)
                continue
        # read infos of images
        while 1:
            try:
                # 获取图像相关信息
                with open(path, 'ab+') as fp:
                    fp.seek(-292, 1)
                    res_ = fp.read(292)
                    res_info = eval(res_.split(b'\x00')[0].decode())
                steel_no_bmp, img_index, cam_no_bmp, steel_start, steel_end, steel_left, steel_right = tuple(res_info.values())
                break
            except Exception as E:
                re_print(E)
                time.sleep(0.001)
                continue
        # 解析数据
        fx = (float(steel_right) - float(steel_left)) / img_w
        fy = (float(steel_end) - float(steel_start)) / img_h
        img_name = '_'.join(['SCILRTB', str(steel_no_bmp), str(cam_no_bmp),
                             str(img_index), str(steel_left), str(steel_start),
                             str(fx), str(fy), 'H']) + str(file_mat)
        dict_info = {'img_gray': img_arr, 'img_path': img_name, 'left_e': l_e, 'right_e': r_e}
        q_size_list = [x.qsize() for x in q_read]
        re_print(f'>>> {name}:{q_size_list} <<<')
        q_list_index = q_size_list.index(min(q_size_list))
        q_read[q_list_index].put(dict_info)


def thread_reads_schema(paths, edge_shift, bin_thres, cam_res,q,logger_):
    for path in paths:
        img_arr = cv2.imread(path, 0)
        l_e, r_e = select_edge(img_arr, edge_shift, bin_thres, cam_res)
        dict_info = {'img_gray': img_arr, 'img_path': path, 'left_e': l_e, 'right_e': r_e}
        q_size_list = [x.qsize() for x in q]
        q_list_index = q_size_list.index(min(q_size_list))
        q[q_list_index].put(dict_info)
        remove_file(path, logger=logger_)


def get_steelno_data(curr_seq, last_seq, is_up_seq, sub_dirs, img_index_dict, q_read, max_steel_save_no, shift, bin_th,
                     cam_res, loop_no, log_oper, cold_start, curr_img_num, total_time,pro_number):
    if is_up_seq:
        seq_num = last_seq
    else:
        seq_num = curr_seq

    dir_index = os.path.join(str(seq_num % int(max_steel_save_no)), '2d'.upper())
    dirs_path_ = [os.path.join(path_, dir_index) for path_ in sub_dirs]

    total_imgs = []
    for dir_path in dirs_path_:
        files = []
        json_path = os.path.join(dir_path, 'record.json')
        while 1:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_info = json.load(f)
                    break
            except Exception as E:
                re_print(E)
                time.sleep(0.001)
                continue

        img_num, cam_no = int(json_info['imgNum']), str(json_info['camNo'])
        if not loop_no:
            files = [os.path.join(dir_path, f'{i + 1}.bmp') for i in range(img_index_dict['imgIndex'][cam_no], img_num)]
        elif loop_no and not cold_start:
            files = [os.path.join(dir_path, f'{(i + 1) % int(loop_no)}.bmp') for i in range(img_index_dict['imgIndex'][cam_no], img_num)]

        total_imgs += files
        img_index_dict['imgIndex'][cam_no] = img_num
    if len(total_imgs):
        t1 = time.time()
        flag_pro_no = pro_number+2
        num = math.ceil(len(total_imgs)/flag_pro_no)
        list_th = []
        for i in range(flag_pro_no):
            th_re = Thread(target=thread_reads,args=(total_imgs[i*num:(i+1)*num],shift, bin_th, cam_res, q_read,f'thread-{i}',))
            th_re.start()
            list_th.append(th_re)
        for th in list_th:
            th.join()
        # 协程
        # tasks = []
        # for path in total_imgs:
        #     tasks.append(gevent.spawn(event_reads,path,shift, bin_th, cam_res, q_read, curr_img_num))
        #
        # gevent.joinall(tasks)
        curr_img_num += len(total_imgs)
        t2 = time.time()
        total_time += t2 - t1
        re_print(f'已读判流水号{seq_num}共 [{curr_img_num}] 张图像平均耗时：{total_time / curr_img_num}s,当前批次读判图像共 【{len(total_imgs)}】 张耗时：{t2 - t1}s')
    else:
        cam_index_info = img_index_dict['imgIndex']
        if is_up_seq:
            log_oper.info(f'流水号{last_seq}各相机图像{cam_index_info}已处理完成,开始处理流水号为{curr_seq}的图像')
            for i in list(img_index_dict['imgIndex'].keys()):
                img_index_dict['imgIndex'][i] = 0
            last_seq = int(curr_seq)
            curr_img_num = 0
            total_time = 0
        else:
            re_print(f'流水号{curr_seq}各相机号图像{cam_index_info}已读取完成,暂时没有待读取图像，等待')
            time.sleep(0.05)

    return curr_seq, last_seq, img_index_dict, curr_img_num, total_time


def get_steelno_data_backup(curr_seq,last_seq,is_up_seq,sub_dirs,img_index_dict,q_read,max_steel_save_no,shift,bin_th,cam_res,loop_no,log_oper,cold_start,curr_img_num,total_time):

    if is_up_seq:
        seq_num = last_seq
    else:
        seq_num = curr_seq

    dir_index = os.path.join(str(seq_num % int(max_steel_save_no)), '2d'.upper())
    dirs_path_ = [os.path.join(path_, dir_index) for path_ in sub_dirs]

    if curr_img_num ==0:
        total_time = 0
    total_imgs = []
    for dir_path in dirs_path_:
        files = []
        json_path = os.path.join(dir_path, 'record.json')
        while 1:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_info = json.load(f)
                    break
            except Exception as E:
                re_print(E)
                continue
        img_num, cam_no = int(json_info['imgNum']), str(json_info['camNo'])
        if not loop_no:
            files = [os.path.join(dir_path, f'{i + 1}.bmp') for i in range(img_index_dict['imgIndex'][cam_no], img_num)]
        elif loop_no and not cold_start:
            files = [os.path.join(dir_path, f'{(i + 1) % int(loop_no)}.bmp') for i in range(img_index_dict['imgIndex'][cam_no], img_num)]

        total_imgs += files
        img_index_dict['imgIndex'][cam_no] = img_num
    if len(total_imgs):
        for path in total_imgs:

            t1 = time.time()
            # 获取图片数组
            _, file_mat = os.path.splitext(os.path.basename(path))
            # read images
            while 1:
                try:
                    img_arr = cv2.imread(path,0)
                    img_h, img_w = img_arr.shape
                    l_e, r_e = select_edge(img_arr, shift, bin_th, cam_res)
                    break
                except Exception as E:
                    re_print(E)
                    continue
            # read infos of images
            while 1:
                try:
                    # 获取图像相关信息
                    with open(path, 'ab+') as fp:
                        fp.seek(-292, 1)
                        res_ = fp.read(292)
                        res_info = eval(res_.split(b'\x00')[0].decode())
                    steel_no_bmp, img_index, cam_no_bmp, steel_start, steel_end, steel_left, steel_right = tuple(res_info.values())
                    break
                except Exception as E:
                    re_print(E)
                    continue
            t1_2 = time.time()
            # 解析数据
            fx = (float(steel_right) - float(steel_left)) / img_w
            fy = (float(steel_end) - float(steel_start)) / img_h
            img_name = '_'.join(['SCILRTB', str(steel_no_bmp), str(cam_no_bmp),
                                 str(img_index), str(steel_left), str(steel_start),
                                 str(fx), str(fy), 'H']) + str(file_mat)
            dict_info = {'img_gray': img_arr, 'img_path': img_name, 'left_e':l_e, 'right_e':r_e}
            q_list_index = curr_img_num % len(q_read)
            q_read[q_list_index].put(dict_info)
            t2 = time.time()
            total_time += t2-t1
            curr_img_num += 1
            re_print(f'图像前处理平均耗时：{total_time/curr_img_num}s,当前耗时：{t2 - t1}s')

    else:
        cam_index_info = img_index_dict['imgIndex']
        if is_up_seq:
            log_oper.info(f'流水号{last_seq}各相机图像{cam_index_info}已处理完成,开始处理流水号为{curr_seq}的图像')
            for i in list(img_index_dict['imgIndex'].keys()):
                img_index_dict['imgIndex'][i] = 0
            last_seq = int(curr_seq)
        else:
            re_print(f'流水号{curr_seq}各相机号图像{cam_index_info}已读取完成,暂时没有待读取图像，等待')
            time.sleep(1)

    return curr_seq,last_seq,img_index_dict,curr_img_num,total_time


def pro_th_readimg(dirs_path, q, schema, loop_num,edge_shift, bin_thres, cam_res, log_path):
    for i in range(len(dirs_path)):
        read_thread = Thread(target=read_images,args=((dirs_path[i],), q, schema, loop_num,edge_shift,bin_thres,cam_res,log_path))
        read_thread.start()


def read_images(dirs_path, q, schema, loop_num,edge_shift, bin_thres, cam_res, log_path, pro_no):
    try:
        while 1:
            try:
                with open('config.yaml', 'r', encoding='utf-8') as f:
                    cg = yaml.load(f.read(), Loader=yaml.FullLoader)
                    break
            except Exception as e:
                time.sleep(0.01)
                continue
        db_config = cg['detect']['dbInfo']
        max_steelno_save = cg['detect']['infoParameter']['max_save_steelno']
        logger_ = LOGS(log_path)
        curr_img_index = {'imgIndex': {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}}
        last_steel_no = -1
        if not schema:
            db_op = DbMysqlOp(ip=db_config['db_ip'], user=db_config['db_user'], psd=db_config['db_psd'],
                              db_name=db_config['db_name'])
        #用于统计
        index_q_num = 0
        total_time = 0
        while True:
            # 有无算法测试程序，也可用来进行离线调试
            if schema:
                total_images_s = []
                for dir_path in dirs_path:
                    files = sorted(glob.glob(os.path.join(dir_path, '*.*')))
                    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
                    total_images_s += images

                if len(total_images_s):
                    flag_no = pro_no+2
                    num = math.ceil(len(total_images_s) / flag_no)
                    list_th = []
                    start_time = time.time()
                    for i in range(flag_no):
                        # paths, edge_shift, bin_thres, cam_res,q,logger_
                        th_re = Thread(target=thread_reads_schema, args=(total_images_s[i * num:(i + 1) * num], edge_shift, bin_thres, cam_res,q,logger_,))
                        th_re.start()
                        list_th.append(th_re)

                    for th in list_th:
                        th.join()
                    end_time = time.time()
                    re_print(f'当前共读取 {len(total_images_s)} 张图像耗时{end_time-start_time}s,平均均耗时{(end_time-start_time)/len(total_images_s)}s')
                else:
                    time.sleep(1)
                    re_print(f'{dirs_path}暂时没有数据了，等待新数据')
            else:
                sql = 'SELECT * FROM steelrecord ORDER BY ID DESC LIMIT 0,1'
                curr_steel_no = db_op.ss_latest_one(sql)
                if curr_steel_no is None:
                    re_print('未检索到任何卷号记录，请检查采集系统是否正常运行')
                    time.sleep(5)
                    continue
                else:
                    curr_steel_no = curr_steel_no[1]
                if int(curr_steel_no) != last_steel_no:
                    # process strat
                    if last_steel_no < 0:
                        cold_boot = True
                        last_steel_no = int(curr_steel_no)

                    # process other time
                    else:
                        re_print(f'》》》 {curr_img_index} 《《《')
                        cold_boot = False
                        curr_steel_no, last_steel_no, curr_img_index,index_q_num, total_time = get_steelno_data(curr_steel_no, last_steel_no,
                                                                                        True, dirs_path,
                                                                                        curr_img_index, q, max_steelno_save,
                                                                                        edge_shift, bin_thres, cam_res,
                                                                                        loop_num, logger_, cold_boot,
                                                                                        index_q_num, total_time, pro_no)

                # 当前卷处理
                else:
                    re_print(f'》》》 {curr_img_index} 《《《')
                    curr_steel_no, last_steel_no, curr_img_index,index_q_num, total_time= get_steelno_data(curr_steel_no, last_steel_no,
                                                                                    False, dirs_path,
                                                                                    curr_img_index, q, max_steelno_save,
                                                                                    edge_shift, bin_thres, cam_res,
                                                                                    loop_num, logger_, cold_boot,
                                                                                    index_q_num, total_time, pro_no)
                    cold_boot = False

    except Exception as E:
        try:
            db_op.close_()
        except:
            pass
        logger_.info(f'{E}')
        raise E


def get_steel_edge(q, q_list,schema, edge_shift, bin_thres, cam_res, log_path):
    try:
        logger_ = LOGS(log_path)
        index = 0
        while True:
            if not q.empty():
                img_infos = q.get()
                img_arr_rgb,img_path = img_infos['img_rgb'],img_infos['img_path']
                l_e, r_e = select_edge(img_arr_rgb, edge_shift, bin_thres, cam_resolution=cam_res)
                infos = {'img_rgb':img_arr_rgb,'img_path':img_path,'left_e':l_e, 'right_e':r_e}
                q_index = index % len(q_list)
                q_list[q_index].put(infos)
                index += 1
                if index > 10000:
                    index = 0
    except Exception as E:
        logger_.info(f'{E}')
        raise E


# (img_arr_rgb, cam_resolution, imgsz, stride, device, pt,......)
def data_tensor_infer(q,result_roi_q,model_obj,cam_resolution,img_resize,stride,device,auto,conf_thres,iou_thres,classes,agnostic_nms,max_det,fp16,debug,log_path,save_rois_path):
    try:
        logger_ = LOGS(log_path)
        model_obj.to(device)
        currt_num = 0
        total_time = 0
        total_get_time = 0
        last_seq = -1
        while 1:
            if not q.empty():
                start_time = time.time()
                img_infos = q.get()
                img_arr_rgb, img_path, left_eg, right_eg = tuple(img_infos.values())
                time_get = time.time()
                if len(img_arr_rgb.shape) == 2:
                    img_arr_rgb = cv2.cvtColor(img_arr_rgb,cv2.COLOR_GRAY2RGB)
                elif len(img_arr_rgb.shape) < 2:
                    raise Exception()
                img_ori_one_shape, img_ori_one_scale_tensor, img_cut_two_shape, img_cut_two_scale_tensor, img_cut_bs_shape, img_cut_bs_scale_tensor \
                    = get_input_tensor(img_arr_rgb, cam_resolution, img_resize, stride, device, auto,fp16)
                model_obj.pre_process_detect(img_path,
                                             save_rois_path,
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
                                             debug=debug,
                                            )
                end_time = time.time()
                currt_num += 1
                total_time += end_time-start_time
                total_get_time += time_get-start_time
                if not debug:

                    curr_seq,curr_camno,curr_img_index = img_path.split('_')[1:4]
                    print('')
                    re_print(f'process-{os.getpid()} 已处理流水号为{curr_seq}的{currt_num}张图片,当前处理的是{curr_camno}号相机第<<{int(curr_img_index)+1}>>张图像,平均耗时{total_time / currt_num}s,平均获取图像耗时{total_get_time / currt_num}')
                    if int(curr_seq) != last_seq:
                        currt_num = 0
                        total_time = 0
                        last_seq = int(curr_seq)
                else:
                    re_print(f'process-{os.getpid()} 已处理{currt_num}张图片,平均耗时{total_time / currt_num}s')
                    if currt_num > int(1e6):
                        currt_num =0
                        total_time = 0
            else:
                # re_print(f'process-{os.getpid()}暂时无处理数据,算法处理速度快于图像前处理速度')
                if debug:
                    re_print(f'process-{os.getpid()}暂时无处理数据,算法处理速度快于图像前处理速度')

    except Exception as E:
        logger_.info(E)
        raise E