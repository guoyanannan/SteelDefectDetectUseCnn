import os
import time
import platform
import logging
import tensorflow as tf


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
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
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
                re_print(f'删除{filename}出错：{E}')


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
                        re_print(f'当前{filename}为非定义格式数据,进行移除')
                        break
                    except Exception as E:
                        re_print(E)
                        num += 1
                        time.sleep(0.1)
                        if num > 10:
                            break
        re_print(f'当前时刻目录{dirname}共 {len(file_path_list)} 文件，清理非定义格式数据后剩余数量:{len(dir_path_dst)}')
        return dir_path_dst
    else:
        return []