#coding=utf-8
from __future__ import print_function
import pandas as pd
import numpy as np
from keras.models import Sequential,load_model
import keras.backend as K
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
import os,time,pickle
from runNN import init_model

def normAnArray(arr,mean,std):
    return (np.asarray(arr) - mean)/std

# def init_model(featureSize,model_path):
#     model = Sequential()
#     model.add(Dense(512, input_dim=featureSize,kernel_initializer='lecun_uniform'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))

#     model.add(Dense(256,kernel_initializer='lecun_uniform'))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.15))


#     model.add(Dense(128,kernel_initializer='lecun_uniform'))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.05))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error',
#                    optimizer='sgd',
#                    metrics=['mae'])
#     if model_path:
#         model.load_weights(model_path)
#     return model

def load_scale(scale_param_path):
    f = open(scale_param_path)
    mean_array = [float(x) for x in f.readline().strip().split()]
    std_array = [float(x) for x in f.readline().strip().split()]
    return mean_array,std_array
    
def evaluate(data_path,scale_param_path,model_path):
    
    mean,std = load_scale(scale_param_path)
    print('Loading data from',data_path,'..................')
    array = pd.read_csv(data_path, header=None).values
    X = array[:,3:]
    date = array[:,0]
    Y = array[:,1]   #Y值,原始数据
    #y2 = array[:,2]   #Y值,sample3数据

    featureSize = X.shape[-1]
    array_x,array_y = array.shape
    #print(X.shape) 
    for i in range(featureSize):
        #print(X[0][i],mean[i],std[i])
        X[:,i] = normAnArray(X[:,i],mean[i],std[i])
	#print(X[0][i],mean[i],std[i])
	#exit(0)
    #print(X[0][:10])
    #exit(0)
    #print('Loading model from',model_path,'................')
    #model = init_model(featureSize,model_path) 
    model = load_model('./model/000002.SZ.h5')
    print(model.get_weights()[0].reshape(-1)[:10])
    score = model.evaluate(X,Y,verbose=0)
    y_pred = model.predict(X,batch_size=200,verbose=0)
    corr = pd.Series(Y.reshape(-1)).corr(pd.Series(y_pred.reshape(-1)))
    
    return score,corr,date,Y.reshape(-1),y_pred.reshape(-1)

if __name__ == '__main__':
    #code_list = ['000002.SZ','000651.SZ','000858.SZ','002353.SZ','600030.SH','600031.SH','600036.SH','600196.SH']
    code_list = ['000002.SZ']
    f = open('result_all.csv','w')
    f.write('code,L1_err,L2_err,corr') #
    f.write('\n')
    for code in code_list:
        data_path = 'result/'+code   # 要预测的股票数据
        model_path = 'model/'+code+'.h5'   # 模型加载路径
        result_path = 'result/'+code+'.result.csv' # 预测出的结果，csv文件
        scale_param_path = 'data/'+code+'.scale'   # mean,std

        print('Predict the stock',code,'........................................') 
        score,corr,date,y,y_pred= evaluate(data_path,scale_param_path,model_path)	  
        print(y_pred[0])
	exit(0)
        f.write(code)
        f.write(',')
        f.write(str(score[0]))
        f.write(',')
        f.write(str(score[1]))
        f.write(',')
        f.write(str(corr))

        f.write('\n')
        l1_err = np.abs(y-y_pred)
        l2_err = np.power((y-y_pred),2)
        fw = open(result_path,'w')
        fw.write('date,y,y_predict,l1_err,l2_err') #每个文件对应一只票，列名：日期，Y值，预测Y值，sample3的Y值，sample3的预测Y值
        fw.write('\n')
            
        for i in range(len(date)):
            fw.write(str(date[i]))
            fw.write(',')
            fw.write(str(y[i]))
            fw.write(',')
            fw.write(str(y_pred[i]))
            fw.write(',')
            fw.write(str(l1_err[i]))
            fw.write(',')
            fw.write(str(l2_err[i]))
            fw.write('\n')
