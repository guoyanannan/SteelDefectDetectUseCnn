import yaml
import argparse
from threading import Thread
from db_process.utils.get_file_info import LOGS
from db_process.utils.db_mysql import read_defect_from_tabel,get_dbop


def run(
        log_path='arith_logs/classifier_logs',
        cam_num=2,
        db_ip='127.0.0.1',
        db_user='root',
        db_psd='nercar',
        transfers="{'db_name': 'temp', 'camera1': 'tempcam1', 'camera2': 'tempcam2'}",
        using="{'db_name': 'ncdcoldstripdefect', 'camera1': 'camdefect1', 'camera2': 'camdefect2'}",
        ):
    logs_oper = LOGS(log_path)
    # 临时表
    temp_db = eval(transfers)
    try:
        try:
            temp_tables = tuple([temp_db[f'camera{i}'] for i in range(1, cam_num + 1)])
        except Exception as E:
            logs_oper.info(f'临时表数量和相机数不一致，请检查参数 transfers 和 cam_num 设置，Error:{E}')
            raise
        # 正式表
        use_db = eval(using)
        try:
            use_tables = tuple([use_db[f'camera{i}'] for i in range(1, cam_num + 1)])
        except Exception as E:
            logs_oper.info(f'正式表数量和相机数不一致，请检查参数 using 和 cam_num 设置，Error:{E}')
            raise
        for i in range(cam_num):
            th = Thread(target=lambda:read_defect_from_tabel(db_ip,
                                                             db_user,
                                                             db_psd,
                                                             logs_oper,
                                                             temp_tables[i],
                                                             use_tables[i],
                                                             temp_db['db_name'],
                                                             use_db['db_name'],
                                                             )
                        )
            th.start()
    except Exception as E:
        logs_oper.info(E)
        raise


def parse_opt():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cls_config = cg['dbinfoprocess']
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default=cls_config['filePath']['logs'],help='directory where log files reside')
    parser.add_argument('--cam_num', type=int, default=cls_config['infoParameter']['cam_num'], help='number of camera')
    parser.add_argument('--db_ip', type=str, default=cls_config['dbInfo']['db_ip'],help='database ip')
    parser.add_argument('--db_user', type=str, default=cls_config['dbInfo']['db_user'],help='database user')
    parser.add_argument('--db_psd', type=str, default=cls_config['dbInfo']['db_psd'],help='database password')
    parser.add_argument('--transfers',type=str, default=str(cls_config['dbInfo']['transfers']), help='temporary database use')
    parser.add_argument('--using', type=str,default=str(cls_config['dbInfo']['using']), help='Official use of database')
    opt = parser.parse_args()
    return opt