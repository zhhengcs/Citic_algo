import pandas as pd
import numpy as np
from keras.models import Sequential,load_model
import keras.backend as K
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping
import pickle
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
##173863

config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)


def normAnArray(arr):
    lTH= np.percentile(arr,0.05)
    uTH= np.percentile(arr,99.95)
    for i in range(len(arr)):
        if arr[i]<lTH:
            arr[i]=lTH
        if arr[i]>uTH:
            arr[i] = uTH

    mid = np.asarray(arr).mean()
    std = np.asarray(arr).std()

    return (np.asarray(arr) - mid)/std,mid,std


def init_data(code):
    
    data = pd.read_csv(code, header=None)
    sampleSize = len(data)
    #data.to_csv('normdat.csv', encoding='utf-8', index=False);
    array =  data.values
    X = array[:,1:]
    featureSize = X.shape[-1]

    # print(X.shape)
    # exit(0)
    mean_array = np.zeros((featureSize,))
    std_array = np.zeros((featureSize,))

    for i in range(featureSize):
        X[:,i],x,y = normAnArray(X[:,i])
        # print(x,y)
        mean_array[i] = x
        std_array[i] = y

    def dump(mean_array,std_array,code=code):
        assert len(mean_array) == len(std_array)
        f = open(code+'.scale','w')
        for o in mean_array:
            f.write(str(o))
            f.write(' ')
        f.write('\n')
        for o in std_array:
            f.write(str(o))
            f.write(' ')
        f.write('\n')
    # def load_scale(code):
    # f = open(code+'.scale')
    # mean_array = [float(x) for x in f.readline().strip().split()]
    # std_array = [float(x) for x in f.readline().strip().split()]
 #        return mean_array,std_array 
    dump(mean_array,std_array)
    Y = array[:,0]

    X_train = X[0:sampleSize,:]
    Y_train = Y[0:sampleSize]
    X_dev   = X[sampleSize-6000:sampleSize-4000,:]
    Y_dev   = Y[sampleSize-6000:sampleSize-4000]
    X_test  = X[sampleSize-4000:sampleSize,:]
    Y_test  = Y[sampleSize-4000:sampleSize]

    return X_train,Y_train,X_dev,Y_dev,X_test,Y_test,featureSize


def init_model(featureSize,model_path = None):
    model = Sequential()
    
    model.add(Dense(512, input_dim=featureSize,init='lecun_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512,init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))


    model.add(Dense(512,init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.05))

    model.add(Dense(1))

    model.summary()
    sgd = SGD(lr=0.01)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['mae'])
    if model_path:
        model.load_weights(model_path)
    
    return model


if __name__ == '__main__':
    # code_list = ['000002.SZ','000651.SZ','000858.SZ','002353.SZ','600030.SH','600031.SH','600036.SH','600196.SH']
    code_list = ['000002.SZ']
    layers = '6'
    for code in code_list:
        data_path = './data/'+code  #数据路径
        model_path = './model/'+code+'.model'+layers #模型保存路径
        nb_epoch = 200
        batch_size = 256  

        X_train,Y_train,X_dev,Y_dev,X_test,Y_test,featureSize = init_data(data_path)
        
        model = init_model(featureSize)
        
        checkpoint = ModelCheckpoint(filepath=model_path,monitor='loss',
                save_best_only=True,save_weights_only=True,mode='min',period=10)
        early_stop = EarlyStopping(monitor='val_mean_absolute_error',patience=10,mode='auto') 
        callback_lists = [checkpoint]#,early_stop]
        history = model.fit(X_train, Y_train,
                batch_size = batch_size,
                epochs = nb_epoch,
                verbose = 1,
                validation_data = (X_dev, Y_dev),
                callbacks=callback_lists)

        score = model.evaluate(X_test, Y_test, verbose=0)
        model.save_weights(model_path)
        print("Saved model to disk",model_path)
        del model
        print('MSELoss:', score[0])
        print('MAE:', score[1])
