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