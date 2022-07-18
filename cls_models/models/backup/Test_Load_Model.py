from tensorflow.keras.models import model_from_json,load_model
from utiles_train.CustomFunctionsForModel import fn,focal_loss,swish
from utiles_train.NewLrDecay import WarmUpReduceLROnPlateau,WarmUpCosineDecayScheduler
import os
from PIL import Image
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


path = './DataAugTrain/train'
cls_name = []
for dirname in os.listdir(path):
    cls_name.append(dirname)
print(cls_name)

# model_name = './run/2021_03_26_14/steel_model_classifier_final.h5'         #   0.015142715894258939 s   1.9382676344651442s
model_name = './run/2021_03_30_18/stage1_model--12--0.953757--1.000000.h5'   #   0.01236114135155311 s    1.5822260929987981s
if True==os.path.exists(model_name):
    try:
        import h5py
        f = h5py.File(model_name, 'r+')
        del f['optimizer_weights']
        f.close()
        model=load_model(model_name,custom_objects={'fn': fn,'focal_loss_fixed':focal_loss(),'swish':swish,'reduce_lr':[WarmUpReduceLROnPlateau,WarmUpCosineDecayScheduler]})
        # model=load_model(model_name,custom_objects={'fn': fn,'focal_loss_fixed':focal_loss(),'swish':swish,})
        print('已删除optimizer weights')
    except:
        print('无需删除optimizer weights')
        # print('删除optimizer weights')
        # import h5py
        # f = h5py.File(self.model_name, 'r+')
        # del f['optimizer_weights']
        # f.close()
        model=load_model(model_name,custom_objects={'fn': fn,'focal_loss_fixed':focal_loss(),'swish':swish,'reduce_lr':[WarmUpReduceLROnPlateau,WarmUpCosineDecayScheduler]})
        # model=load_model(model_name,custom_objects={'fn': fn,'focal_loss_fixed':focal_loss(),'swish':swish,})


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


    #k = Model.predict(Image_arr,batch_size=1)
    path = './DataAugTrain/validation/凹坑'
    Image_arr = []
    import time
    t1_strat = time.time()
    i = 0
    for filename in os.listdir(path):
        #i +=1
        filename = os.path.join(path,filename)
        print(filename)
        img = Image.open(filename)
        img = img.convert('RGB')
        img = img.resize((299,299))
        img = np.array(img)
        img = img/255
        Image_arr.append(img)
        print(img.shape,img)
    Image_arr = np.array(Image_arr)
    t2_end = time.time()
    tt2 = t2_end-t1_strat
    print(Image_arr.shape)
    print('-----------------------------------------------')

    kk = model.predict(Image_arr,batch_size=Image_arr.shape[0])
    t_strat = time.time()
    k = model.predict(Image_arr,batch_size=Image_arr.shape[0])
    print(k)
    #i = 0
    # t_end = time.time()
    # average_speed = (t_end-t_strat+tt2)/i
    # print(average_speed,'s')


    for kk in k:
        print(os.listdir(path)[i])
        print(kk,type(kk),len(kk))
        print(np.argmax(kk),kk[np.argmax(kk)])
        class_name = cls_name[np.argmax(kk)]
        print(class_name)
        i += 1
        print('--------------------------------------------------------------------------------------------------------------')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
