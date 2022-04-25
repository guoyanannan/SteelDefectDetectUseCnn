# # 作者:郭亚男
# # 单位:北科工研
# # 时间:2022-04-24

from multiprocessing import freeze_support
from detect_yolov5.steel_detect_inference import parse_opt, run


def main():
    opt = parse_opt()
    print(vars(opt))
    run(**vars(opt))


if __name__ == '__main__':
    # exe进程中的限制
    freeze_support()
    main()
