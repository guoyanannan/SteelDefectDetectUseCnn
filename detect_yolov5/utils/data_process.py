import os
import os.path
import cv2
import glob
import torch
import time
import json
import numpy as np
from PIL import Image
from .normaloperation import LOGS
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


def edge_select(img_src, img_rgb,radio,shift,debug=False):
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
    if debug:
        cv2.line(img_rgb,(left_edge,0),(left_edge,img_rgb.shape[0]-1),(255,0,0),3)
        cv2.line(img_rgb,(right_edge,0),(right_edge,img_rgb.shape[0]-1),(255,0,0),3)
        cv2.namedWindow('edge',0)
        cv2.imshow('edge',img_rgb)
    return left_edge,right_edge


def select_edge(img_arr,shift,bin_threshold=0.35,cam_resolution='4k',debug=False):
    img_rgb = img_arr.copy()  # RGB
    img_arr = cv2.cvtColor(img_arr,cv2.COLOR_RGB2GRAY)
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
    if debug:
        cv2.namedWindow('img2threshold',0)
        cv2.imshow('img2threshold', img_th)
    return edge_select(img_th,img_rgb,w_scale,shift)


def numpy_to_torch_tensor(img_arr,img_size,stride,auto,device):
    img_arr = letterbox(img_arr, img_size, stride, auto=auto)[0]
    img_arr = img_arr.transpose((2, 0, 1))
    img_arr = np.ascontiguousarray(img_arr)
    img_arr = torch.from_numpy(img_arr).to(device)
    img_arr = img_arr.float()
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


def get_input_tensor(img_arr,  # img RGB
               cam_resolution,  # 相机分辨率
               img_size,  # 期望尺寸 [h,w]
               stride,    # 网络最终下采样倍数
               device,    # str cuda:0 or 1 or 2,  or str cpu
               auto,
               bin_threshold=0.35,  #
               ):

    img_ori = img_arr  # RGB
    img_split = img_arr.copy()
    if cam_resolution.lower() == '4k'.lower():
        # 原图 - 1张 - 原始形状 - bchw
        img_ori_one_shape = (1, img_ori.shape[-1], img_ori.shape[0], img_ori.shape[1])
        # 原图 - 1张 - 缩放形态 - bchw
        img_ori_one_scale_tensor = numpy_to_torch_tensor(img_ori, img_size, stride, auto, device)

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
                img_cut_one_scale_tensor = numpy_to_torch_tensor(tmp, img_size, stride, auto, device)
                img_cut_list.append(img_cut_one_scale_tensor)

        img_cut_bs_scale_tensor = torch.cat(img_cut_list, dim=0)
        img_cut_bs_shape = img_cut_ori[0]

    if cam_resolution.lower() == '8k'.lower():
        # 原图 - 1张 - 原始形状 - bchw
        img_ori_one_shape = (1, img_ori.shape[-1], img_ori.shape[0], img_ori.shape[1])
        # 原图 - 1张 - 缩放形态 - bchw
        img_ori_one_scale_tensor = numpy_to_torch_tensor(img_ori, img_size, stride, auto, device)

        # 原图 - 2张 - 原始形状 - bchw
        img_cut_two_shape = (2, img_ori.shape[-1], img_ori.shape[0], img_ori.shape[1] // 2)
        # 原图 - 2张 - 缩放形态 - bchw
        img_two_list = []
        img_cut_two_scale_tensor_1 = numpy_to_torch_tensor(img_ori[:, :img_ori.shape[1] // 2], img_size, stride, auto,
                                                           device)
        img_cut_two_scale_tensor_2 = numpy_to_torch_tensor(img_ori[:, img_ori.shape[1] // 2:], img_size, stride, auto,
                                                           device)
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
                img_cut_one_scale_tensor = numpy_to_torch_tensor(tmp, img_size, stride, auto, device)
                img_cut_list.append(img_cut_one_scale_tensor)

        img_cut_bs_scale_tensor = torch.cat(img_cut_list)
        img_cut_bs_shape = img_cut_ori[0]

    return \
           img_ori_one_shape, img_ori_one_scale_tensor,\
           img_cut_two_shape, img_cut_two_scale_tensor, \
           img_cut_bs_shape, img_cut_bs_scale_tensor


def get_steelno_data(curr_seq,last_seq,is_up_seq,sub_dirs,img_index_dict,q_read):

    if is_up_seq:
        seq_num = last_seq
    else:
        seq_num = curr_seq

    dir_index = os.path.join(str(seq_num % 2), '2d'.upper())
    dirs_path_ = [os.path.join(path_, dir_index) for path_ in sub_dirs]
    total_imgs = []
    for dir_path in dirs_path_:
        json_path = os.path.join(dir_path, 'record.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            json_info = json.load(f)
        img_num, cam_no = int(json_info['imgNum']), str(json_info['camNo'])
        files = [os.path.join(dir_path, f'{i + 1}.bmp') for i in range(img_index_dict['imgIndex'][cam_no], img_num)]
        total_imgs += files
        img_index_dict['imgIndex'][cam_no] = img_num
    if len(total_imgs):
        for path in total_imgs:
            # 获取图片数组
            _, file_mat = os.path.basename(path).split('.')
            img_arr = np.array(Image.open(path), dtype=np.uint8)
            img_arr_rgb = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
            img_h, img_w = img_arr_rgb.shape[:-1]
            # 获取当前图片信息
            with open(path, 'ab+') as fp:
                fp.seek(-292, 1)
                res_ = fp.read(292)
                res_info = eval(res_.split(b'\x00')[0].decode())
            steel_no_bmp, img_index, cam_no_bmp, steel_start, steel_end, steel_left, steel_right = tuple(
                res_info.values())
            fx = (float(steel_right) - float(steel_left)) / img_w
            fy = (float(steel_end) - float(steel_start)) / img_h
            img_name = '_'.join(['SCILRTB', str(steel_no_bmp), str(cam_no_bmp),
                                 str(img_index), str(steel_left), str(steel_start),
                                 str(fx), str(fy), 'H']) + f'.{file_mat}'
            print(img_name)
            dict_info = {'img_rgb': img_arr_rgb, 'img_path': img_name}
            q_read.put(dict_info)
    else:
        cam_index_info = img_index_dict['imgIndex']
        if is_up_seq:
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}>>>>>>>>流水号{last_seq}各相机号已处理{cam_index_info},准备处理当前{curr_seq}的数据....................')
            for i in list(img_index_dict['imgIndex'].keys()):
                img_index_dict['imgIndex'][i] = 0
            last_seq = int(curr_seq)
        else:
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}>>>>>>>>流水号{curr_seq}各相机号已处理{cam_index_info},暂时没有待处理数据，等待....................')
            time.sleep(1)

    return curr_seq,last_seq,img_index_dict


def read_images(dirs_path, q, schema, log_path):
    try:
        logger = LOGS(log_path)
        curr_img_index = {'imgIndex':{'1': 0,'2': 0,'3':0,'4': 0,'5': 0,'6':0,'7': 0,'8': 0,'9':0,'10':0}}
        last_steel_no = -1
        db_op = DbMysqlOp(ip='localhost', user='root', psd='nercar', db_name='ncdcoldstrip')
        while True:
            if schema:
                for dir_path in dirs_path:
                    files = sorted(glob.glob(os.path.join(dir_path, '*.*')))
                    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
                    if len(images):
                        for path in images:
                            img_arr = np.array(Image.open(path), dtype=np.uint8)
                            img_arr_rgb = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
                            # file_name = os.path.basename(path)
                            dict_info = {'img_rgb':img_arr_rgb, 'img_path': path}
                            q.put(dict_info)
                            remove_file(path, logger=logger)
                    else:
                        time.sleep(1)
                        logger.info(f'{dir_path}暂时没有数据了，等待新数据....................')
            else:

                sql = 'SELECT * FROM steelrecord ORDER BY ID DESC LIMIT 0,1'
                curr_steel_no = db_op.ss_latest_one(sql)[1]

                print('当前流水号：',curr_steel_no)
                print('上一次流水号：',last_steel_no)
                if int(curr_steel_no) != last_steel_no:
                    # process strat
                    if last_steel_no <0:
                        last_steel_no = int(curr_steel_no)

                    # process other time
                    else:
                        time.sleep(0.5)
                        curr_steel_no, last_steel_no, curr_img_index = get_steelno_data(curr_steel_no, last_steel_no, True, dirs_path,
                                                                                        curr_img_index, q)
                # 当前卷处理
                else:
                    curr_steel_no, last_steel_no, curr_img_index = get_steelno_data(curr_steel_no, last_steel_no, False, dirs_path,
                                                                                    curr_img_index, q)

    except Exception as E:
        print('E:',E)
        db_op.close_()
        logger.info(f'{E}')
        raise E


def get_steel_edge(q, q_list,schema, edge_shift, bin_thres, cam_res, log_path):
    logger = LOGS(log_path)
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

        else:
            time.sleep(0.01)
            print(f'数据队列暂时没有新数据，等待....................')