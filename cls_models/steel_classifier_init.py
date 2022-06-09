import time

import numpy as np
from PIL import Image
from cls_models.utils.get_file_info import ReadConvertFile
from cls_models.models.mobilenet_v3 import MobileNetV3Large
from cls_models.models.xception import Xception
from cls_models.models.inception_v3 import InceptionV3
from cls_models.models.efficientnet import EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5
from cls_models.models.resnet_common import ResNeXt50,ResNet50
from cls_models.models.nasnet import NASNetLarge,NASNet
from cls_models.utils.common_oper import re_print,delete_batch_file
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class ClassificationAlgorithm(ReadConvertFile):
    def __init__(self,**kwargs):
        super(ClassificationAlgorithm, self).__init__(**kwargs)
        self.bs = kwargs['batch_size']
        self.model_path = kwargs['model_path']
        self.dynamic_size = kwargs['dynamic_size']
        self.op_log = kwargs['op_log']
        self.read_ini()
        self.read_xml()
        self.create_model()

    def img_proces(self, batch_img_path):
        image_arr = []
        image_list = []
        for filename in batch_img_path:
            try:
                img = Image.open(filename)
            except:
                delete_batch_file([filename])
                continue
            image_list.append(np.array(img))
            img = img.convert('RGB')
            img = img.resize((self.imgsize, self.imgsize))
            img = np.array(img)
            img = img / 255
            image_arr.append(img)
        image_arr = np.array(image_arr)
        re_print(f'当前批次输入形状:[{image_arr.shape}],缓存数量:[{len(image_list)}]')
        delete_batch_file(batch_img_path)
        if image_arr is not None:
            return image_arr,image_list
        else:
            self.op_log.info(f'当前批次共{len(batch_img_path)}张图片数据，有效数量为 0 请检查!')
            raise

    def create_model(self):
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

        if self.dynamic_size:
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
            self.model = Model(inputs=base_model.input, outputs=predictions)
        else:
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            predictions = Dense(self.classnum, activation='softmax')(x)
            self.model = Model(inputs=base_model.input, outputs=predictions)
        try:
            load_status = self.model.load_weights(self.model_path)
            test_arr = np.zeros((1,self.imgsize,self.imgsize,3))
            _ = self.model.predict(test_arr,batch_size=1)
            self.op_log.info(f'{self.ModelName}模型加载成功')
        except Exception as EEer:
            self.op_log.info('模型加载失败，错误信息：{}'.format(EEer))
            raise

    def inference(self, batch_path):
        v1 = time.time()
        img_arr_bs,img_list = self.img_proces(batch_path)
        v2 = time.time()
        results = self.model.predict(img_arr_bs, batch_size=img_arr_bs.shape[0])
        v3 = time.time()
        scores = results.max(axis=1).astype('float16') * 100
        inter_no = [self.index_and_inter[i] for i in results.argmax(axis=1)]
        cls_name = [self.inter_and_name[i] for i in inter_no]
        enter_no = [self.inter_and_exter[i] for i in inter_no]
        total_result_iter = zip(batch_path,img_list,cls_name,inter_no,enter_no,scores)
        v4 = time.time()
        re_print(f'读取加模型预测加结果整理共耗时：{v4-v1}s,读取{v2-v1}s,推理{v3-v2}s,整理{v4-v3}s')
        return total_result_iter

    def inference_asyn(self, batch_path,img_arr_bs,img_list_bs):

        print(img_arr_bs.shape)
        results = self.model.predict(img_arr_bs, batch_size=img_arr_bs.shape[0])
        scores = results.max(axis=1).astype('float16') * 100
        inter_no = [self.index_and_inter[i] for i in results.argmax(axis=1)]
        cls_name = [self.inter_and_name[i] for i in inter_no]
        enter_no = [self.inter_and_exter[i] for i in inter_no]
        total_result_iter = zip(batch_path,img_list_bs,cls_name,inter_no,enter_no,scores)

        return total_result_iter



