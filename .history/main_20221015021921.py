
import numpy as np
import torch
from python_segy.get_patch import*  
from python_segy.gain import *

DOWNLOAD = False

# original data generates patch
train_data = datagenerator("/home/astroc/Projects/SEGY",patch_size = (128,128),
                           stride = (32,32), train_data_num = float('inf'), 
                           download=DOWNLOAD,datasets=1,aug_times=0,scales = [1],
                           verbose=True,jump=1,agc=True)