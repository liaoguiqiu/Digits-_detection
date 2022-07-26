import json as JSON
import cv2
import math
import numpy as np
import os
import random 
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from pathlib import Path

class EXCEL_saver(object):
    def __init__(self, num = 3 ):
        #self.dir = "D:/Deep learning/out/1out_img/Ori_seg_rec_Unet/"
        self.plots = np.zeros((1,num))  
        self.plots[0,:] = np.arange(num)
        self.save_cnt = 0
        #self.firstflag = False
    def append_save(self,vector,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.plots = np.append(self.plots, [vector], axis=0)
        if (self.save_cnt % 10 == 0):  # each 10 filter
            self.plots[:,1] = signal.medfilt(self.plots[:,1] ,3)
            self.plots[:, 1] = signal.medfilt(self.plots[:, 1], 3)
            a = 1.0
            b=[0.25,0.25,0.25,0.25]
            self.plots[:, 2] = signal.lfilter(b,a, self.plots[:,1])
            pass
        DF1 = pd.DataFrame(self.plots)
        DF1.to_csv(save_dir+"error_buff.csv")
        self.save_cnt = self.save_cnt + 1
