import tensorflow as tf
import os
from PIL import Image
import numpy as np
from TOOLS.steel_classifier import Classifier
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) >= 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_name = r'D:\1\steel_model\steel_model_classifier.h5'   #   0.01236114135155311 s    1.5822260929987981s
if True==os.path.exists(model_name):

    classifier = Classifier(224, 224, model_name, 'EfficientNetB2', 3)
    classifier.LoadModel()


# try:
#Model = load_model('./run/2021_03_26_14/steel_model_classifier_final.h5',custom_objects={'fn': single_class_accuracy,'focal_loss_fixed':focal_loss(),'swish':swish})
# except:
#     print('删除optimizer weights')
#     import h5py
#     f = h5py.File('./run/2021_03_26_14/steel_model_classifier_final.h5', 'r+')
#     del f['optimizer_weights']
#     f.close()
#     Model = load_model('./run/2021_03_26_14/steel_model_classifier_final.h5',
#                        custom_objects={'fn': single_class_accuracy(),
#                                        'focal_loss_fixed': focal_loss(),
#                                        'swish': swish()})

    while True:
        #k = Model.predict(Image_arr,batch_size=1)
        path = './ArNTPreClassifiedImage'
        Image_arr = []
        import time
        t1_strat = time.time()
        i = 0
        for filename in os.listdir(path):
            #i +=1
            filename = os.path.join(path,filename)
            #print(filename)
            img = Image.open(filename)
            img = img.convert('RGB')
            img = img.resize((299,299))
            img = np.array(img)
            img = img/255
            Image_arr.append(img)
            #print(img.shape,img)
        Image_arr = np.array(Image_arr)
        t2_end = time.time()
        tt2 = t2_end-t1_strat
        print(Image_arr.shape)
        print('处理数据时间:',tt2)

        #kk = model.predict(Image_arr,batch_size=Image_arr.shape[0])
        t_strat = time.time()
        k = classifier.PredictImages(Image_arr)
        t_end = time.time()
        print(k)
        print('模型推理时间:', t_end-t_strat)
        print(f"{Image_arr.shape[0]}张图像共耗时{t_end-t1_strat}s")
    #i = 0
    # t_end = time.time()
    # average_speed = (t_end-t_strat+tt2)/i
    # print(average_speed,'s')


    # for kk in k:
    #     print(os.listdir(path)[i])
    #     print(kk,type(kk),len(kk))
    #     print(np.argmax(kk),kk[np.argmax(kk)])
    #     class_name = cls_name[np.argmax(kk)]
    #     print(class_name)
    #     i += 1
    #     print('--------------------------------------------------------------------------------------------------------------')
    #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
