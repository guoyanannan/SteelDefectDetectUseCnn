from models.mobilenet_v3 import MobileNetV3Large
from models.xception import Xception
from models.inception_v3 import InceptionV3
from models.densenet import DenseNet121
from models.efficientnet import EfficientNetB2
from models.resnet_common import ResNeXt50
from models.nasnet import NASNetLarge,NASNet
import numpy as np
import os
import tensorflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

image = np.random.randn(1,224,224,3).astype('float16')
print(image)
print(image.dtype)





# base_model = NASNet(weights=None,include_top=False,input_shape=(224,224,3))
# base_model = ResNeXt50(weights=None,include_top=False,)
# base_model = EfficientNetB2(weights=None,include_top=False,)
# base_model = MobileNetV3Large(weights=None,include_top=False,)
# base_model = DenseNet121(weights=None,include_top=False,)
base_model = Xception(weights=None,include_top=False,
                      backend=tensorflow.keras.backend,
                      layers=tensorflow.keras.layers,
                      models=tensorflow.keras.models,
                      utils=tensorflow.keras.utils
                      )
x1 = GlobalAveragePooling2D()(base_model.output)
print(x1)
print(x1.shape)
print(x1.shape[-1])
# # x = Dense(1024,activation='relu')(x1)
# # predictions = Dense(5,activation='softmax')(x)
# # model = Model(inputs=base_model.input,outputs=predictions)
x = Reshape((1, 1,-1))(x1)
print(x)
x = Conv2D(1024,
          kernel_size=1,
          padding='same',
          name='Conv_2')(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Conv2D(5,
          kernel_size=1,
          padding='same',
          name='Logits')(x)
x = Flatten()(x)
x = Softmax(name='Predictions/Softmax')(x)
model = Model(inputs=base_model.input,outputs=x)
model.summary()
#
rusult = model.predict(image,batch_size=image.shape[0])
print(base_model.layers)
print(base_model.output)
print(x1)
print(rusult)
print(rusult.shape)
print(rusult.dtype)
rusult = np.reshape(rusult,(-1,5))
rusult = np.array([[0.2 ,0.3 ,0.2, 0.5, 0.2],[0.2 ,0.3 ,0.2, 0.7, 0.2]])
print(rusult.dtype)
L = np.where(rusult==rusult.max())[1]
print(L)

# if __name__ == '__main__':
#     print('此脚本主要用来验证，解决输入尺寸无法改变大小问题！！！！！，input指定形状会将图结构的特征图尺寸固定！！！')