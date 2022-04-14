import os
import re
import sys
import logging
import platform
import torch
import datetime
from torchvision.ops import nms

try:
    import codecs
except ImportError:
    codecs = None


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

class MultiprocessHandler(logging.FileHandler):
    """支持多进程的TimedRotatingFileHandler"""
    def __init__(self,dir_path,when='D',backupCount=0,encoding=None,delay=False):
        """
        dir_path 日志文件存储路径
        when 时间间隔的单位
        backupCount 保留文件个数
        delay 是否开启 OutSteam缓存
        True 表示开启缓存
        OutStream输出到缓存，待缓存区满后，刷新缓存区，并输出缓存数据到文件。
        False表示不缓存
        OutStrea直接输出到文件
        """
        self.logDirPath = dir_path
        self.backupCount = backupCount
        self.when = when.upper()
        # 正则匹配 年-月-日
        self.extMath = r"^\d{4}-\d{2}-\d{2}"

        # S 每秒建立一个新文件
        # M 每分钟建立一个新文件
        # H 每天建立一个新文件
        # D 每天建立一个新文件
        self.when_dict = {
            'S': "%Y-%m-%d-%H-%M-%S.log",
            'M': "%Y-%m-%d-%H-%M.log",
            'H': "%Y-%m-%d-%H.log",
            'D': "%Y-%m-%d.log"
        }
        # 日志文件日期格式
        self.log_name = self.when_dict.get(self.when)
        if not self.log_name:
            raise ValueError(u"指定的日期间隔单位无效: %s" % self.when)
        #拼接文件路径 格式化字符串
        self.filefmt = os.path.join(self.logDirPath,self.log_name)
        #使用当前时间，格式化文件格式化字符串
        self.filePath = datetime.datetime.now().strftime(self.filefmt)

        if codecs is None:
            encoding = None

        logging.FileHandler.__init__(self,self.filePath,'a+',encoding,delay)

    def shouldChangeFileToWrite(self):
        """更改日志写入目的写入文件
        :return True 表示已更改，False 表示未更改"""
        # 以当前时间获得新日志文件路径
        _filePath = datetime.datetime.now().strftime(self.filefmt)
        # 新日志文件日期 不等于 旧日志文件日期，则表示 已经到了日志切分的时候
        #   更换日志写入目的为新日志文件。
        # 例如 按 天 （D）来切分日志
        #   当前新日志日期等于旧日志日期，则表示在同一天内，还不到日志切分的时候
        #   当前新日志日期不等于旧日志日期，则表示不在
        #   同一天内，进行日志切分，将日志内容写入新日志内。
        if _filePath != self.filePath:
            self.filePath = _filePath
            return True
        return False

    def doChangeFile(self):
        """输出信息到日志文件，并删除多于保留个数的所有日志文件"""
        #日志文件的绝对路径
        self.baseFilename = os.path.abspath(self.filePath)
        #stream == OutStream
        #stream is not None 表示 OutStream中还有未输出完的缓存数据
        if self.stream:
            #flush close 都会刷新缓冲区，flush不会关闭stream，close则关闭stream
            #self.stream.flush()
            self.stream.close()
            #关闭stream后必须重新设置stream为None，否则会造成对已关闭文件进行IO操作。
            self.stream = None
        # delay 为False 表示 不OutStream不缓存数据 直接输出
        #   所有，只需要关闭OutStream即可
        if not self.delay:
            #这个地方如果关闭colse那么就会造成进程往已关闭的文件中写数据，从而造成IO错误
            #delay == False 表示的就是 不缓存直接写入磁盘
            #我们需要重新在打开一次stream
            #self.stream.close()
            self.stream = self._open()
        #删除多于保留个数的所有日志文件
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                try:
                    os.remove(s)
                except:
                    pass

    def getFilesToDelete(self):
        """获得过期需要删除的日志文件"""
        #分离出日志文件夹绝对路径
        #split返回一个元组（absFilePath,fileName)
        #例如：split('I:\ScripPython\char4\mybook\util\logs\mylog.2017-03-19）
        #返回（I:\ScripPython\char4\mybook\util\logs， mylog.2017-03-19）
        # _ 表示占位符，没什么实际意义，
        dirName,log_file_name = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        #print('dirname:',dirName)
        for fileName in fileNames:
            log_file_time = log_file_name.split('.')[0]
            #print('time:',log_file_time)
            #匹配符合规则的日志文件，添加到result列表中
            if re.compile(self.extMath).match(log_file_time):
                result.append(os.path.join(dirName,fileName))
        result.sort()

        #返回  待删除的日志文件
        #   多于 保留文件个数 backupCount的所有前面的日志文件。
        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result

    def emit(self, record):
        """发送一个日志记录
        覆盖FileHandler中的emit方法，logging会自动调用此方法"""
        try:
            if self.shouldChangeFileToWrite():
                self.doChangeFile()
            logging.FileHandler.emit(self,record)
        except (KeyboardInterrupt,SystemExit):
            raise
        except:
            self.handleError(record)

class LOGS(object):
    def __init__(self, dir_path, when='D',record_num=0):
        # 输出日志路径
        self.get_when = when
        self.num_backup = record_num
        dir_path = os.path.join(dir_path, '')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # 设置日志格式#和时间格式
        FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
        DATEFMT = '%Y-%m-%d %H:%M:%S'
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
        self.logger.addHandler(self.get_file_handler(dir_path))
        self.logger.addHandler(self.get_console_handler())
        # 设置日志的默认级别
        self.logger.setLevel(logging.INFO)

    # 输出到文件handler的函数定义
    def get_file_handler(self, filename):
        filehandler = MultiprocessHandler(filename, when=self.get_when, encoding="utf-8",backupCount=self.num_backup)
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

