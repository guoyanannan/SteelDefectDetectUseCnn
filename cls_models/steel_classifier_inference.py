import os
import time
import yaml
import math
import argparse
from cls_models.utils.get_file_info import LOGS
from cls_models.utils.db_mysql import get_camdefect_no,check_temp_db,get_dbop,write_defect_to_table
from cls_models.steel_classifier_init import ClassificationAlgorithm
from cls_models.utils.common_oper import select_device, select_no_img,parse_data,re_print


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
                defect_cam_num[str(i)]=max(temp_defect_no_info[str(i)],defect_no_info[str(i)])
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

        while True:
            if not debug:
                db_oper_ = get_dbop(db_ip,db_user,db_psd,temp_db['db_name'],logs_oper)
            file_name_list = sorted(os.listdir(rois_dir), key=lambda x: os.path.getmtime(os.path.join(rois_dir, x)))
            files_path = [os.path.join(rois_dir, fileName) for fileName in file_name_list]
            files_path = select_no_img(files_path, os.path.basename(rois_dir))
            if files_path:
                num_bs = math.ceil(len(files_path)/bs)
                for i in range(num_bs):
                    batch_img_path = files_path[i * bs:(i + 1) * bs]
                    batch_result_iter = classifier_model.inference(batch_img_path)
                    defect_cam_num, db_defects_info = parse_data(batch_result_iter,
                                                                 save_intercls,
                                                                 offline_result,
                                                                 defect_cam_num,
                                                                 negative,
                                                                 ignore,
                                                                 curr_schema
                                                                 )
                    if debug:
                        num = 0
                        for info in db_defects_info:
                            num += len(info)
                        re_print(f'采用非数据库形式存储当前批次共{num}条信息: {db_defects_info}')
                    else:
                        write_defect_to_table(db_oper_,db_defects_info,temp_tables)
            else:
                if bool(db_oper_):
                    db_oper_.close_()
                re_print(f'目录[{os.path.basename(rois_dir)}]暂时没有数据了,等待')
                time.sleep(1)
    except Exception as E:
        logs_oper.info(E)
        raise Exception(f'{E}')


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