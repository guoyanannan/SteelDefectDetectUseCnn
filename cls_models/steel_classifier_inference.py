import os
import sys
import time
import yaml
import argparse

from threading import Thread
from queue import Queue
from cls_models.utils.get_file_info import LOGS
from cls_models.utils.data_load import thread_load_data
from cls_models.utils.db_mysql import get_camdefect_no,check_temp_db,get_dbop,write_defect_to_table
from cls_models.steel_classifier_init import ClassificationAlgorithm
from cls_models.utils.common_oper import select_device, select_no_img,parse_data,re_print,delete_dir


def run(
        weights='steel_model_classifier.h5',
        ini='steel_classifier_interaction.ini',
        xml='steel_class_table.xml',
        log_path='arith_logs/classifier_logs',
        rois_dir='result_roi_8k',
        save_intercls=(),
        offline_result='classification_result',
        schema="{'flag': 0, 'camera1': 'F:/defectImg1', 'camera2': 'F:/defectImg2'}",
        bs=32,
        negative=0,
        ignore=50,
        cam_num=2,
        device='0',
        dynamic_size=False,
        debug=False,
        db_ip='127.0.0.1',
        db_user='root',
        db_psd='nercar',
        transfers="{'db_name': 'temp', 'camera1': 'tempcam1', 'camera2': 'tempcam2'}",
        using="{'db_name': 'ncdcoldstripdefect', 'camera1': 'camdefect1', 'camera2': 'camdefect2'}",
        ):

    logs_oper = LOGS(log_path)
    ss = select_device(device)
    logs_oper.info(ss)

    if not debug:
        if os.path.exists(rois_dir):
            delete_dir(rois_dir)
        else:
            os.makedirs(rois_dir)
    try:
        curr_schema = eval(schema)
        try:
            cam_defect_dir = tuple([curr_schema[f'camera{i}'] for i in range(1, cam_num + 1)])
        except Exception as E:
            logs_oper.info(f'相机路径数量和相机数不一致，请检查参数 schema 和 cam_num 设置，Error:{E}')
            raise
        if not debug:
            # 查看临时库和表是否存在，不存在就创建
            temp_db = eval(transfers)
            try:
                temp_tables = tuple([temp_db[f'camera{i}'] for i in range(1,cam_num+1)])
            except Exception as E:
                logs_oper.info(f'临时表数量和相机数不一致，请检查参数 transfers 和 cam_num 设置，Error:{E}')
                raise
            check_temp_db(db_ip, db_user, db_psd, temp_db['db_name'], temp_tables,logs_oper)
            # 查询临时表里面的缺陷数
            temp_defect_no_info = get_camdefect_no(db_ip, db_user, db_psd, temp_db['db_name'], temp_tables,logs_oper)
            # 查询正式表里面的缺陷数
            use_db = eval(using)
            try:
                use_tables = tuple([use_db[f'camera{i}'] for i in range(1,cam_num+1)])
            except Exception as E:
                logs_oper.info(f'正式表数量和相机数不一致，请检查参数 using 和 cam_num 设置，Error:{E}')
                raise
            defect_no_info = get_camdefect_no(db_ip,db_user,db_psd,use_db['db_name'],use_tables,logs_oper)
            # 以记录最大为开始计数
            defect_cam_num = {}
            for i in range(1,len(defect_no_info)+1):
                defect_cam_num[str(i)] = max(temp_defect_no_info[str(i)],defect_no_info[str(i)])
            logs_oper.info(f'程序启动时各相机缺陷数量{defect_cam_num}')
        else:
            defect_cam_num = {}
            for i in range(1,cam_num+1):
                defect_cam_num[str(i)] = 0

        classifier_model = ClassificationAlgorithm(xml_path=xml,
                                                   ini_path=ini,
                                                   batch_size=bs,
                                                   model_path=weights,
                                                   dynamic_size=dynamic_size,
                                                   op_log=logs_oper)

        img_size = classifier_model.imgsize
        read_q = Queue(3)
        # 数据线程
        th = Thread(target=lambda:thread_load_data(read_q,rois_dir,bs,img_size,logs_oper),)
        th.start()

        # 模型线程
        thread_process_model_res(db_ip, db_user, db_psd,temp_db['db_name'],
                                 read_q,classifier_model,save_intercls,offline_result,
                                 defect_cam_num,negative,ignore,temp_tables,
                                 curr_schema,logs_oper,debug)

    except Exception as E:
        logs_oper.info(E)
        raise Exception(f'{E}')


def thread_process_model_res(db_ip, db_user, db_psd,db_name,
                             index_q,model,save_intercls,offline_result,
                             defect_cam_num,negative,ignore,temp_tables,
                             curr_schema,logger,debug):
    total_time = 0
    get_q_data = 0
    get_model_res = 0
    get_pro_res = 0
    write_tabel = 0
    num_total = 0
    num_cur = 0
    while True:
        if not debug:
            db_oper_ = get_dbop(db_ip, db_user, db_psd, db_name, logger)
        if not index_q.empty():
            num_cur += 1
            v1 = time.time()
            image_arr, image_list, batch_img_path = index_q.get()
            num_total += len(batch_img_path)
            v2 = time.time()
            batch_result_iter = model.inference_asyn(batch_img_path, image_arr, image_list)
            v3 = time.time()
            defect_cam_num, db_defects_info = parse_data(batch_result_iter,
                                                         save_intercls,
                                                         offline_result,
                                                         defect_cam_num,
                                                         negative,
                                                         ignore,
                                                         curr_schema,
                                                         debug
                                                         )

            v4 = time.time()
            if debug:
                num = 0
                for info in db_defects_info:
                    num += len(info)
                re_print(f'采用非数据库形式存储当前批次共{num}条信息: {db_defects_info}')
            else:
                v5 = time.time()
                write_defect_to_table(db_oper_, db_defects_info, temp_tables)
                v6 = time.time()
                get_q_data += v2 - v1
                get_model_res += v3 - v2
                get_pro_res += v4 - v3
                write_tabel += v6 - v5
                total_time += v6 - v1
                re_print(
                    f'当前队列数量{index_q.qsize()} ,此批次平均共耗时：{total_time / num_cur}s,读取{get_q_data / num_cur}s,推理{get_model_res / num_cur}s,FPS {num_total / get_model_res}，'
                    f'存入共享加整理{get_pro_res / num_cur}s, 数据库写入{write_tabel / num_cur}s')
                print('--' * 40)
        else:
            if bool(db_oper_):
                db_oper_.close_()
            re_print(f'缺陷队列中暂时没有数据了,等待')
            time.sleep(1)


def parse_opt():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cls_config = cg['classifier']
    # print(cls_config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=cls_config['filePath']['weights'], help='model path(s)')
    parser.add_argument('--ini', type=str, default=cls_config['filePath']['ini'], help='ini path(s)')
    parser.add_argument('--xml', type=str, default=cls_config['filePath']['xml'], help='xml path(s)')
    parser.add_argument('--log_path', type=str, default=cls_config['filePath']['logs'], help='directory where log files reside')
    parser.add_argument('--rois_dir', type=str, default=cls_config['filePath']['input'], help='directory where input images reside')
    parser.add_argument('--save_intercls', type=tuple, default=cls_config['filePath']['internal_number'], help='internal number to save')
    parser.add_argument('--offline_result', type=str, default=cls_config['filePath']['offline_result'], help='directory where outupt images reside')
    parser.add_argument('--schema', type=str,default=str(cls_config['filePath']['schema']), help='the architecture type')
    parser.add_argument('--bs', type=int, default=cls_config['infoParameter']['bs'], help='the number of input images')
    parser.add_argument('--negative', type=int, default=cls_config['infoParameter']['negative'], help='negative class')
    parser.add_argument('--ignore', type=int, default=cls_config['infoParameter']['score_ignore'], help='class threshold')
    parser.add_argument('--cam_num', type=int, default=cls_config['infoParameter']['cam_num'], help='number of camera')
    parser.add_argument('--device', type=str, default=cls_config['infoParameter']['device'], help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dynamic_size', action='store_true', default=cls_config['infoParameter']['dynamicsize'],help='use dynamic Size to inference')
    parser.add_argument('--debug', action='store_true', default=cls_config['infoParameter']['debug'], help='is or not debug mode')
    parser.add_argument('--db_ip', type=str, default=cls_config['dbInfo']['db_ip'],help='database ip')
    parser.add_argument('--db_user', type=str, default=cls_config['dbInfo']['db_user'],help='database user')
    parser.add_argument('--db_psd', type=str, default=cls_config['dbInfo']['db_psd'],help='database password')
    parser.add_argument('--transfers',type=str, default=str(cls_config['dbInfo']['transfers']), help='temporary database use')
    parser.add_argument('--using', type=str,default=str(cls_config['dbInfo']['using']), help='Official use of database')
    opt = parser.parse_args()
    return opt