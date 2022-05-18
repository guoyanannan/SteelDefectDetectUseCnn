import os,sys,yaml
import configparser
import shutil
import time
import math
import numpy as np
import tensorflow.keras
import matplotlib.pyplot as plt
from models.mobilenet_v3 import MobileNetV3Large
from models.xception import Xception
from models.inception_v3 import InceptionV3
from models.densenet import DenseNet121
from models.efficientnet import EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5
from models.resnet_common import ResNeXt50,ResNet50
from models.nasnet import NASNetLarge,NASNet
from xml.etree import ElementTree as ET
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf

class ReadConfigFile(object):
    __flag = None
    def __new__(cls, *args, **kwargs):
        if not cls.__flag:
            cls.__flag = object.__new__(cls)
            return cls.__flag
        else:
            return cls.__flag


    def __init__(self,**kwargs):
        self.xml_path = kwargs['xml_path']
        self.ini_path = kwargs['ini_path']

    # 读类别表
    def read_xml(self):
        '''
        path:xml文件路径
        return:内部编号：外部编号
        '''
        if True == os.path.exists(self.xml_path):
            per = ET.parse(self.xml_path)
            p = per.findall('./缺陷类别/类别总数')
            classnum = p[0].text
            self.inter_and_exter = {}
            self.inter_and_name = {}
            for i in range(int(classnum)):
                s = './缺陷类别/类别%d' % i
                # print(s)
                p = per.findall(s)
                for oneper in p:
                    interal_no = ""
                    interal_name = ""
                    exteral_no = ""
                    # for child in oneper.getchildren():
                    for child in list(oneper):
                        # print(child.tag,':',child.text)
                        if child.tag == "内部编号":
                            interal_no = child.text
                        if child.tag == "名称":
                            interal_name = child.text
                        if child.tag == "外部编号":
                            exteral_no = child.text
                    self.inter_and_exter[interal_no] = exteral_no
                    self.inter_and_name[interal_no] = interal_name

    def read_ini(self):
        if True == os.path.exists(self.ini_path):
            conf = configparser.ConfigParser()
            conf.read(self.ini_path)
            sections = conf.sections()
            self.index_and_inter = {}
            self.classnum = int(conf.get("Classifier", "ClassNum"))
            self.imgsize = int(conf.get("Classifier", "ImgSize"))
            self.ModelName = str(conf.get("Classifier", "Model"))
            for i in range(int(self.classnum)):
                classno = "Class%d" % i
                interal_no = conf.get("ClassConversion", classno)
                self.index_and_inter[i] = interal_no


class ClassificationAlgorithm(ReadConfigFile):
    # __flag = None
    def __new__(cls, *args, **kwargs):
        return super(ClassificationAlgorithm, cls).__new__(cls)

    def __init__(self,**kwargs):
        super(ClassificationAlgorithm, self).__init__(**kwargs)
        self.img_path = kwargs['img_path']
        self.img_save_path = kwargs['img_save_path']
        self.bs = kwargs['batch_size']
        self.model_path = kwargs['model_path']
        self.Mdt = kwargs['Mdt']
        self.is_move = kwargs['move_file']
        self.read_ini()
        self.read_xml()
        self.check_dir()

    def check_dir(self):
        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)
        else:
            for filename in os.listdir(self.img_save_path):
                filepath = os.path.join(self.img_save_path,filename)
                try:
                    os.remove(filepath)
                except Exception as E:
                    print(f'删除{filename}出错：{E}')

    def select_no_img(self,dir_path_list):
        if dir_path_list:
            dir_path_dst = dir_path_list.copy()
            ImgMat = ['bmp', 'jpg', 'BMP', 'JPG', 'png']
            for filepath in dir_path_list:
                filename = filepath.split('\\')[-1]
                if filename.split('.')[-1] not in ImgMat:
                    num = 1
                    while True:
                        try:
                            os.remove(filepath)
                            dir_path_dst.remove(filepath)
                            print(f'当前{filename}为非定义格式数据,以进行移除!!!!!')
                            break
                        except Exception as E:
                            print(E)
                            num += 1
                            time.sleep(1)
                            if num >10:
                                break
            print(f'当前时刻目录{self.img_path}-数量:{len(dir_path_list)}文件已清洗完毕，剩余数量:{len(dir_path_dst)}！！！！')
            return dir_path_dst

    def Imageproces(self,batch_img_path):
        Image_arr = []
        for filename in batch_img_path:
            img = Image.open(filename)
            img = img.convert('RGB')
            img = img.resize((self.imgsize, self.imgsize))
            img = np.array(img)
            img = img / 255
            Image_arr.append(img)
        Image_arr = np.array(Image_arr)
        print('Image_arr_shape:',Image_arr.shape)
        if Image_arr is not None:
            return Image_arr
        else:
            print('读取数据为空！！！！！！！')
            return None

    def createModel(self,Mdt):
        if self.ModelName in ('inception_v3','InceptionV3'):
            base_model = InceptionV3(weights=None, include_top=False)
            x = base_model.output
            # x = base_model.layers[196].output
        elif self.ModelName == "ResNet50":
            base_model = ResNet50(weights=None, include_top=False)
            x = base_model.output
        elif self.ModelName in ("efficient_d2", 'EfficientNetB2'):
            base_model = EfficientNetB2(weights=None, include_top=False)
            x = base_model.output
        elif self.ModelName in ("efficient_d3", 'EfficientNetB3'):
            base_model = EfficientNetB3(weights=None, include_top=False)
            x = base_model.output
        elif self.ModelName in ("efficient_d4", 'EfficientNetB4'):
            base_model = EfficientNetB4(weights=None, include_top=False)
            x = base_model.output
        elif self.ModelName in ("efficient_d5", 'EfficientNetB5'):
            base_model = EfficientNetB5(weights=None, include_top=False)
            x = base_model.output
        elif self.ModelName == "Xception":
            base_model = Xception(weights=None, include_top=False)
            x = base_model.output
        elif self.ModelName == "MobileNetV3Large":
            base_model = MobileNetV3Large(weights=None, include_top=False)
            x = base_model.output
        elif self.ModelName == "NASNet":
            base_model = NASNet(weights=None, include_top=False, input_shape=(self.imgsize, self.imgsize, 3))
            x = base_model.output

        if Mdt:
            x = GlobalAveragePooling2D()(x)
            x = Reshape((1, 1, -1))(x)
            x = Conv2D(1024,
                       kernel_size=1,
                       padding='same',
                       name='Conv_2')(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = Conv2D(self.classnum,
                       kernel_size=1,
                       padding='same',
                       name='Logits')(x)
            x = Flatten()(x)
            predictions = Softmax(name='Predictions/Softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        else:
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            predictions = Dense(self.classnum, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        try:
            load_status = model.load_weights(self.model_path)
            model.summary()
            print('模型加载成功！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')
            return model

        except Exception as EEer:
            print('模型加载失败，错误信息：{}'.format(EEer))
            return None

    def ClsInference(self):
        img_path = self.img_path
        bs = self.bs
        model = self.createModel(self.Mdt)
        while True:
            filePath = [os.path.join(img_path,fileName) for fileName in os.listdir(img_path)]
            filePath = self.select_no_img(filePath)
            if filePath:
                num_bs = math.ceil(len(filePath)/bs)
                for i in range(num_bs):
                    batch_img_path = filePath[i*bs:(i+1)*bs]
                    bs_img = self.Imageproces(batch_img_path)
                    if bs_img is not None:
                        result = model.predict(bs_img,batch_size=bs_img.shape[0])
                        result_max = result.max(axis=1).astype('float16')
                        result_name = [self.inter_and_name[self.index_and_inter[i]] for i in result.argmax(axis=1)]
                        batch_img_save_path = [result_name[i] +"_"+str(int(result_max[i]*100))+"_"+str(j.split('\\')[1]) for i,j in enumerate(batch_img_path)]
                        for j in range(len(batch_img_path)):
                            if self.is_move:
                                shutil.move(batch_img_path[j],os.path.join(self.img_save_path,batch_img_save_path[j]))
                            else:
                                shutil.copy(batch_img_path[j], os.path.join(self.img_save_path, batch_img_save_path[j]))
                        print(f'当前已完成{i+1}批次数据的解析存储！！！！！')

            else:
                time.sleep(1)
                print('没有需要分类的数据了！！！！！！！！')

def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cg = yaml.load(f.read(), Loader=yaml.FullLoader)
    use_cuda = cg['infors']['cuda']
    if not use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    classifier_op = ClassificationAlgorithm(img_path=cg['filepath']['img_path'],
                                            model_path=cg['filepath']['model_path'],
                                            xml_path=cg['filepath']['xml_path'],
                                            ini_path=cg['filepath']['ini_path'],
                                            img_save_path=cg['filepath']['img_save_path'],
                                            batch_size=cg['infors']['batch_size'],
                                            Mdt=cg['infors']['DynamicSizing'],
                                            move_file= cg['infors']['move'],
                                            )

    classifier_op.ClsInference()

if __name__ == '__main__':
    main()
    # with open('config.yaml', 'r', encoding='utf-8') as f:
    #     cg = yaml.load(f.read(), Loader=yaml.FullLoader)
    #
    # use_cuda = cg['infors']['cuda']
    # if not use_cuda:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # else:
    #     physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #     assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #
    # classifier_op = ClassificationAlgorithm(img_path=cg['filepath']['img_path'],
    #                                         model_path=cg['filepath']['model_path'],
    #                                         xml_path=cg['filepath']['xml_path'],
    #                                         ini_path=cg['filepath']['ini_path'],
    #                                         img_save_path=cg['filepath']['img_save_path'],
    #                                         batch_size=cg['infors']['batch_size'],
    #                                         Mdt=cg['infors']['DynamicSizing'],
    #                                         move_file= cg['infors']['move'],
    #                                         )
    #
    # classifier_op.ClsInference()
