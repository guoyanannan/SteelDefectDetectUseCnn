import os.path
import cv2
import glob
import torch
import time
import numpy as np
from PIL import Image


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
    for i in range(10):
        try:
            os.remove(file_path)
            is_remv = True
            break
        except:
            time.sleep(0.01)
            continue
    if not is_remv:
        logger.info(f'{file_path} deletion failed and the next loop will be entered')


# def load_images(dir_path_list,
#                 logger,
#                 read_queue_list,
#                 device,
#                 img_size=416,
#                 stride=64,
#                 cam_resolution='4k',
#                 auto=True,
#                 schema=True,  # 有无算法测试程序
#                 ):
#
#     for dir_path in dir_path_list:
#         if os.path.exists(dir_path):
#             continue
#         else:
#             logger.info(f'{dir_path} is does not exist. Please check!')
#             raise Exception()
#
#     while True:
#         img_path_list = []
#         if schema:
#             get_img_list_1 = time.time()
#             for dir_path in dir_path_list:
#                 files = sorted(glob.glob(os.path.join(dir_path, '*.*')))  # dir
#                 images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
#                 img_path_list += images
#             get_img_list_2 = time.time()
#
#             print(f'数据列表整合阶段:{get_img_list_2-get_img_list_1}s')
#             if len(img_path_list):
#                 for i in range(len(img_path_list)):
#                     start_get_img_1 = time.time()
#                     path_img = img_path_list[i]
#                     img_ori = cv2.imread(path_img)  # BGR HWC
#                     img_draw = img_ori.copy()
#                     img_split = img_ori.copy()
#                     end_get_img_1 = time.time()
#                     print(f'数据读取时间和copy时间:{end_get_img_1-start_get_img_1}s')
#                     if cam_resolution.lower() == '4k'.lower():
#                         # 寻边
#                         start_select_edge = time.time()
#                         left_edge,right_edge = select_edge(path_img, -10, 0.35, cam_resolution=cam_resolution)
#                         end_select_edge = time.time()
#                         print(f'寻边界：{end_select_edge-start_select_edge}s')
#
#                         procee_data_time_1 = time.time()
#                         # 原图 - 1张 - 原始形状 - bchw
#                         img_ori_one_shape = (1,img_ori.shape[-1],img_ori.shape[0],img_ori.shape[1])
#                         # 原图 - 1张 - 缩放形态 - bchw
#                         img_ori_one_scale_tensor = numpy_to_torch_tensor(img_ori,img_size,stride,auto,device)
#
#                         # 对于4k图像来说，不会切分两张，在此是为了和8k的统一加入队列的标准
#                         # 原图 - 2张 - 原始形状 - bchw
#                         img_cut_two_shape = img_ori_one_shape
#                         # 原图 - 2张 - 缩放形态 - bchw
#                         img_cut_two_scale_tensor = img_ori_one_scale_tensor
#
#                         # 切图
#                         win_stride = 768
#                         flag_temp = True
#                         img_cut_list = []
#                         img_cut_ori = []
#                         for r in range(0, (img_split.shape[0] - img_split.shape[0]) + 1, win_stride):  # H方向进行切分
#                             for c in range(0, (img_split.shape[1] - 1024) + 1, win_stride):  # W方向进行切分
#                                 tmp = img_split[r: r + img_split.shape[0], c: c + 1024]
#                                 # 切图 - 原始形状 - bchw
#                                 if flag_temp:
#                                     img_cut_bs_shape_ = (1,tmp.shape[-1],tmp.shape[0],tmp.shape[1])
#                                     img_cut_ori.append(img_cut_bs_shape_)
#                                     flag_temp = False
#                                 # 切图 - 缩放形态 - bchw
#                                 img_cut_one_scale_tensor = numpy_to_torch_tensor(tmp,img_size,stride,auto,device)
#                                 img_cut_list.append(img_cut_one_scale_tensor)
#
#                         img_cut_bs_scale_tensor = torch.cat(img_cut_list,dim=0)
#                         img_cut_bs_shape = img_cut_ori[0]
#
#                         procee_data_time_2 = time.time()
#                         print(f'数据规整时间：{procee_data_time_2-procee_data_time_1}')
#                     if cam_resolution.lower() == '8k'.lower():
#                         # 寻边
#                         left_edge, right_edge = select_edge(path_img, -10, 0.35, cam_resolution=cam_resolution)
#
#                         # 原图 - 1张 - 原始形状 - bchw
#                         img_ori_one_shape = (1, img_ori.shape[-1], img_ori.shape[0], img_ori.shape[1])
#                         # 原图 - 1张 - 缩放形态 - bchw
#                         img_ori_one_scale_tensor = numpy_to_torch_tensor(img_ori, img_size, stride, auto, device)
#
#                         # 原图 - 2张 - 原始形状 - bchw
#                         img_cut_two_shape = (2, img_ori.shape[-1], img_ori.shape[0], img_ori.shape[1]//2)
#                         # 原图 - 2张 - 缩放形态 - bchw
#                         img_two_list = []
#                         img_cut_two_scale_tensor_1 = numpy_to_torch_tensor(img_ori[:, :img_ori.shape[1]//2], img_size, stride, auto, device)
#                         img_cut_two_scale_tensor_2 = numpy_to_torch_tensor(img_ori[:, img_ori.shape[1]//2:], img_size, stride, auto, device)
#                         img_two_list +=[img_cut_two_scale_tensor_1,img_cut_two_scale_tensor_2]
#                         img_cut_two_scale_tensor = torch.cat(img_two_list,dim=0)
#                         # 切图
#                         win_stride = 896
#                         flag_temp = True
#                         img_cut_list = []
#                         img_cut_ori = []
#                         for r in range(0, (img_split.shape[0] - img_split.shape[0]) + 1, win_stride):  # H方向进行切分
#                             for c in range(0, (img_split.shape[1] - 1024) + 1, win_stride):  # W方向进行切分
#                                 tmp = img_split[r: r + img_split.shape[0], c: c + 1024]
#                                 # 切图 - 原始形状 - bchw
#                                 if flag_temp:
#                                     img_cut_bs_shape_ = (1, tmp.shape[-1], tmp.shape[0], tmp.shape[1])
#                                     img_cut_ori.append(img_cut_bs_shape_)
#                                     flag_temp = False
#                                 # 切图 - 缩放形态 - bchw
#                                 img_cut_one_scale_tensor = numpy_to_torch_tensor(tmp, img_size, stride, auto, device)
#                                 img_cut_list.append(img_cut_one_scale_tensor)
#
#                         img_cut_bs_scale_tensor = torch.cat(img_cut_list)
#                         img_cut_bs_shape = img_cut_ori[0]
#
#                     dict_inf0 = {'img_path': path_img,
#                                  'img_draw': img_draw,
#                                  'left_edge': left_edge,
#                                  'right_edge': right_edge,
#                                  'img_ori_one_shape': img_ori_one_shape,
#                                  'img_ori_one_scale_tensor': img_ori_one_scale_tensor,
#                                  'img_cut_two_shape': img_cut_two_shape,
#                                  'img_cut_two_scale_tensor': img_cut_two_scale_tensor,
#                                  'img_cut_bs_shape': img_cut_bs_shape,
#                                  'img_cut_bs_scale_tensor': img_cut_bs_scale_tensor,
#                                  'cam_resolution': cam_resolution.lower(),
#                                  }
#                     queue_num = len(read_queue_list)
#                     queue_index = i % queue_num
#                     if read_queue_list[queue_index].full() is False:
#                         read_queue_list[queue_index].put(dict_inf0)
#                         # 添加完成后进行删除
#                         remove_file(path_img,logger=logger)
#                     else:
#                         logger.info(f'data_queue_{queue_index+1} is full,image add failed and he next loop will be entered!')
#
#             else:
#                 logger.info('The current folder is empty, waiting for a new image to be written！')
#                 time.sleep(2)
#                 continue


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


def read_images(dirs_path, q, schema, logger):
    curr_seq = 0
    last_seq = -100
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
                        #remove_file(path,logger=logger)
        else:
            # dir_path (path1,path2),几个相机写几个,地址需至包含 0 1目录
            # 读物info.json文件获取当前流水号，图像数量
            # 如果当前流水号 != 上一次流水号：循环图像读取，结束后上一次流水号=当前流水号

            # 读取图片里信息，获得 top bottom left right 真实坐标，换算fx，fy，

            #
            pass


def get_steel_edge(q, q_list,schema, edge_shift, bin_thres, cam_res, logger):

    while True:
        index = 0
        if schema:
            if not q.empty():
                img_infos = q.get()
                index += 1
                img_arr_rgb,img_path = img_infos['img_rgb'],img_infos['img_path']
                l_e, r_e = select_edge(img_arr_rgb, edge_shift, bin_thres, cam_resolution=cam_res)
                infos = {'img_rgb':img_arr_rgb,'img_path':img_path,'left_e':l_e, 'right_e':r_e}
                q_index = index % len(q_list)
                q_list[q_index].put(infos)

            else:
                time.sleep(0.1)
                logger.info(f'暂时没有新数据，等待！！！！！！！！！！！！！！！！')

        else:
            pass