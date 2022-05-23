import os
import re
import sys
import datetime
import logging
try:
    import codecs
except ImportError:
    codecs = None


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


