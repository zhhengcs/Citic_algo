#coding=utf-8
#from __future__ import print_function
import os
import os.path as osp
import argparse

import tensorflow as tf

from keras.models import load_model
from keras import backend as K
import numpy as np
import h5py
# code_list = ['000002.SZ','000651.SZ','000858.SZ','002353.SZ','600030.SH','600031.SH','600036.SH','600196.SH']
code_list = ['000002.SZ']
for code in code_list:    
    fw = open(code+'.weight','w')
    model = load_model(code+'.h5')
    weight = model.get_weights()
    for w in weight:
        if len(w.shape) == 2:
            w = w.transpose()
        w = w.reshape(-1)
        w = w.tolist()
      
        stri = ' '.join(map(str,w))
        fw.write(stri)
        fw.write('\n')


