import yaml
import argparse
import os
from cls_models.utils.get_file_info import LOGS
from cls_models.steel_classifier_init import ClassificationAlgorithm
from cls_models.utils.common_oper import select_device, select_no_img


def run(
        weights='steel_model_classifier.h5',
        ini='steel_classifier_interaction.ini',
        xml='steel_class_table.xml',
        log_path='arith_logs/classifier_logs',
        rois_dir='result_roi_8k',
        offline_result='classification_result',
        schema="{'flag': 0, 'camera1': 'F:/defectImg1', 'camera2': 'F:/defectImg2'}",
        bs=32,
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
    clssifer_model = ClassificationAlgorithm(xml_path=xml,
                                             ini_path=ini,
                                             batch_size=bs,
                                             model_path=weights,
                                             dynamic_size=dynamic_size,
                                             op_log=logs_oper)
    files_path = [os.path.join(rois_dir, fileName) for fileName in os.listdir(rois_dir)]
    files_path = select_no_img(files_path,os.path.basename(rois_dir))
    if files_path:
        pass
    # print(weights)
    # print(ini)
    # print(xml)
    # print(log_path)
    # print(rois_dir)
    # print(offline_result)
    # print(schema)
    # print(bs)
    # print(device)
    # print(dynamic_size)
    # print(debug)
    # print(db_ip)
    # print(db_user)
    # print(db_psd)
    # print(transfers)
    # print(using)


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
    parser.add_argument('--offline_result', type=str, default=cls_config['filePath']['offline_result'], help='directory where outupt images reside')
    parser.add_argument('--schema', type=str,default=str(cls_config['filePath']['schema']), help='the architecture type')
    parser.add_argument('--bs', type=int, default=cls_config['infoParameter']['bs'], help='the number of input images')
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