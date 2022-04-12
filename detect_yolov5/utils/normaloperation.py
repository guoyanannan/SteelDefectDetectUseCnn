import os
import sys
import logging
import platform
import torch
from torchvision.ops import nms
from time import strftime


def xyxy_img_nms(boxes,classnames,iou_thresh):
    if len(boxes) != 0:
        result_ = []
        for cls_index in range(len(classnames)):
            the_boxes = boxes[torch.where(boxes[:, -1] == cls_index)[0], :]
            if len(the_boxes):
                res_index = nms(the_boxes[:,:4],the_boxes[:,4],iou_thresh)
                result_.append(torch.index_select(the_boxes,0,res_index))
        if len(result_):
            result = torch.vstack(result_)
            return result
        else:
            return None
    else:
        raise Exception('The input value is null, and what is needed is a tensor of shape (n,6)')


def marge_box(pre_boxes,class_names,iou_thresh=0.0,debug=False):
    #用于框的融合
    bboxes_arr = pre_boxes
    if debug:
        print('去除空列表原始后的result数组:',len(bboxes_arr))
    if len(bboxes_arr) != 0 :
        cls_boxes_after_merge = []
        for every_cls_index in range(len(class_names)):
            #  left, top, right, bottom, score,cls
            the_boxes = bboxes_arr[torch.where(bboxes_arr[:,-1] == every_cls_index)[0], :]
            while len(the_boxes) > 1:
                if debug:
                    print(f'类别{every_cls_index}的框:\n{the_boxes}')
                x1 = the_boxes[:, 0]
                y1 = the_boxes[:, 1]
                x2 = the_boxes[:, 2]
                y2 = the_boxes[:, 3]
                areas = (x2 - x1 + 1) * (y2 - y1 + 1)
                if debug:
                    print('results中的每个框的面积：',areas)
                order = torch.argsort(areas)
                if debug:
                    print('按照面积排序后的box的索引:order',order,type(order))
                index = order[-1]
                x11 = torch.maximum(x1[index], x1[order[:-1]])
                y11 = torch.maximum(y1[index], y1[order[:-1]])
                x22 = torch.minimum(x2[index], x2[order[:-1]])
                y22 = torch.minimum(y2[index], y2[order[:-1]])
                w = torch.maximum(torch.tensor(0.0).to(x11.device), x22 - x11 + 1)
                h = torch.maximum(torch.tensor(0.0).to(x11.device), y22 - y11 + 1)
                intersection = w * h
                # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框
                ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
                left = torch.where(ious > iou_thresh)[0]
                if debug:
                    print('满足符合交并比的oder索引left：\n',left,type(left))
                if len(left) != 0:
                    #获得满足和最大面积框IOU满足阈值的bbox的索引
                    order = order[left]
                    x_1 = torch.min(torch.cat([x1[order],torch.tensor([x1[index]]).to(x1.device)],dim=0))
                    y_1 = torch.min(torch.cat([y1[order],torch.tensor([y1[index]]).to(y1.device)],dim=0))
                    x_2 = torch.max(torch.cat([x2[order],torch.tensor([x2[index]]).to(x2.device)],dim=0))
                    y_2 = torch.max(torch.cat([y2[order],torch.tensor([y2[index]]).to(y2.device)],dim=0))
                    score = torch.tensor(1)
                    The_boxes_after_merge= torch.tensor([[x_1,y_1,x_2,y_2,score,every_cls_index]]).to(the_boxes.device)
                    delete_index = torch.cat([order, torch.tensor([index]).to(order.device)],dim=0).sort()[0]
                    if debug:
                        print('delete_index:',delete_index)
                    for i in range(len(delete_index)):
                        k = int(delete_index[i])-i
                        k = torch.tensor(k).to(the_boxes.device)
                        the_boxes = the_boxes[torch.arange(the_boxes.size()[0]).to(the_boxes.device) != k]
                    if debug:
                        print('删除后框:',the_boxes)
                    the_boxes = torch.cat([the_boxes,The_boxes_after_merge],dim=0)
                    if debug:
                        print(f'融合后框:\n{the_boxes}')
                else:
                    cls_boxes_after_merge.append(the_boxes[order[-1]])
                    the_boxes = the_boxes[torch.arange(the_boxes.size()[0]).to(the_boxes.device) != order[-1]]
            else:
                the_boxes[0][-2] = 1
                cls_boxes_after_merge.append(the_boxes[0])

        marge_end = torch.vstack(cls_boxes_after_merge)
        marge_end[:,-2]= 1
        if debug:
            print('最终融合后的结果:',marge_end)
        return marge_end
    else:
        return None


def select_device(device, log_op, batch_size=0, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'detect torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ[
            'CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)"  # bytes to MB
    else:
        s += 'CPU'

    if not newline:
        s = s.rstrip()
    log_op.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe

    return torch.device('cuda:0' if cuda else 'cpu')


class LOGS(object):
    def __init__(self,path):
        # 输出日志路径
        PATH = os.path.join(path, '')
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        # 设置日志格式#和时间格式
        FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
        DATEFMT = '%Y-%m-%d %H:%M:%S'
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
        self.log_filename = '{0}{1}.log'.format(PATH, strftime("%Y-%m-%d"))

        self.logger.addHandler(self.get_file_handler(self.log_filename))
        self.logger.addHandler(self.get_console_handler())
        # 设置日志的默认级别
        self.logger.setLevel(logging.INFO)

    # 输出到文件handler的函数定义
    def get_file_handler(self, filename):
        filehandler = logging.FileHandler(filename, encoding="utf-8")
        filehandler.setFormatter(self.formatter)
        return filehandler

    # 输出到控制台handler的函数定义
    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def info(self, message: str):
        self.logger.info(str(message))


if __name__ == '__main__':
    my_logg = LOGS('../logs_haha\\')
    select_device('0',my_logg)

