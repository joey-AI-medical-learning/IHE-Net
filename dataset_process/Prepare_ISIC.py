# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019
@author: Reza Azad
"""
import h5py
import numpy as np
import scipy.io as sio
# import scipy.misc as sc
import glob
from imageio.v2 import imread
from PIL import Image

# Parameters
height = 224
width = 224
channels = 3


Dataset_add = 'input/ISIC2018/'
Tr_add = 'ISIC2018_Task1-2_Training_Input'

Tr_list = glob.glob(Dataset_add + Tr_add + '/*.jpg')
print("Tr_list length: {}".format(len(Tr_list)))
# It contains 2000 training samples
Data_train_2017 = np.zeros([2594, height, width, channels])
Label_train_2017 = np.zeros([2594, height, width])

print('Reading ISIC 2017')
for idx in range(len(Tr_list)):
     print(idx + 1)
     img = imread(Tr_list[idx])
     pil_img = Image.fromarray(img)
     img = np.double(pil_img.resize((height, width), Image.BILINEAR))
     Data_train_2017[idx, :, :, :] = img

     b = Tr_list[idx]
     a = b[0:len(Dataset_add)]
     b = b[len(b) - 16: len(b) - 4]
     add = (a + 'ISIC2018_Task1_Training_GroundTruth/' + b + '_segmentation.png')
     img2 = imread(add)
     pil_img2 = Image.fromarray(img2)
     img2 = np.double(pil_img2.resize((height, width), Image.BILINEAR))
     Label_train_2017[idx, :, :] = img2

 print('Reading ISIC 2017 finished')


Train_img = Data_train_2017[0:1838, :, :, :]
Validation_img = Data_train_2017[1838:2594, :, :, :]

Train_mask = Label_train_2017[0:1838, :, :]
Validation_mask = Label_train_2017[1838:2594, :, :]

np.save('input/ISIC2018-1838-756/data_train', Train_img)
np.save('input/ISIC2018-1838-756/data_val', Validation_img)

np.save('input/ISIC2018-1838-756/mask_train', Train_mask)
np.save('input/ISIC2018-1838-756/mask_val', Validation_mask)



























# # #######################################################ISIC2017 train val text = 2000 150 600########################################
# #
# Dataset_add = 'input/ISIC2018/'
# Tr_add = 'ISIC2018_Task1-2_Training_Input'
#
# Tr_list = glob.glob(Dataset_add + Tr_add + '/*.jpg')
# print("Tr_list length: {}".format(len(Tr_list)))
# # It contains 2000 training samples
# Data_train_2017 = np.zeros([2594, height, width, channels])
# Label_train_2017 = np.zeros([2594, height, width])
#
# print('Reading ISIC 2017')
# for idx in range(len(Tr_list)):
#     print(idx + 1)
#     img = imread(Tr_list[idx])
#     pil_img = Image.fromarray(img)
#     img = np.double(pil_img.resize((height, width), Image.BILINEAR))
#     Data_train_2017[idx, :, :, :] = img
#
#     b = Tr_list[idx]
#     a = b[0:len(Dataset_add)]
#     b = b[len(b) - 16: len(b) - 4]
#     add = (a + 'ISIC2018_Task1_Training_GroundTruth/' + b + '_segmentation.png')
#     img2 = imread(add)
#     pil_img2 = Image.fromarray(img2)
#     img2 = np.double(pil_img2.resize((height, width), Image.BILINEAR))
#     Label_train_2017[idx, :, :] = img2
#
# print('Reading ISIC 2017 finished')
#
#
# Train_img = Data_train_2017[0:1816, :, :, :]
# Validation_img = Data_train_2017[1816:1816+259, :, :, :]
# Test_img = Data_train_2017[1816+259:2594, :, :, :]
#
# Train_mask = Label_train_2017[0:1816, :, :]
# Validation_mask = Label_train_2017[1816:1816+259, :, :]
# Test_mask = Label_train_2017[1816+259:2594, :, :]
#
# np.save('data_train', Train_img)
# np.save('data_test', Test_img)
# np.save('data_val', Validation_img)
#
# np.save('mask_train', Train_mask)
# np.save('mask_test', Test_mask)
# np.save('mask_val', Validation_mask)


#######################################################ISIC2017 train val text = 2000 150 600########################################

# Dataset_add = 'input/ISIC2017/'
# Tr_add = 'ISIC-2017_Validation_Data'
#
# Tr_list = glob.glob(Dataset_add + Tr_add + '/*.jpg')
# print("Tr_list length: {}".format(len(Tr_list)))
# # It contains 2000 training samples
# Data_train_2017 = np.zeros([150, height, width, channels])
# Label_train_2017 = np.zeros([150, height, width])
#
# print('Reading ISIC 2018')
# for idx in range(len(Tr_list)):
#     print(idx + 1)
#     img = imread(Tr_list[idx])
#     pil_img = Image.fromarray(img)
#     img = np.double(pil_img.resize((height, width), Image.BILINEAR))
#     Data_train_2017[idx, :, :, :] = img
#
#     b = Tr_list[idx]
#     a = b[0:len(Dataset_add)]
#     b = b[len(b) - 16: len(b) - 4]
#     add = (a + 'ISIC-2017_Validation_Part1_GroundTruth/' + b + '_segmentation.png')
#     img2 = imread(add)
#     pil_img2 = Image.fromarray(img2)
#     img2 = np.double(pil_img2.resize((height, width), Image.BILINEAR))
#     Label_train_2017[idx, :, :] = img2
#
# print('Reading ISIC 2018 finished')
#
# np.save('input/ISIC2017-2000-150-600/data_val', Data_train_2017)
# np.save('input/ISIC2017-2000-150-600/mask_val', Label_train_2017)


###########################################################################################################
#
# Dataset_add = 'input/ISIC2018/'
# Tr_add = 'ISIC2018_Task1-2_Training_Input'
#
# Tr_list = glob.glob(Dataset_add + Tr_add + '/*.jpg')
# print("Tr_list length: {}".format(len(Tr_list)))
# Data_train_2017 = np.zeros([2594, height, width, channels])
# Label_train_2017 = np.zeros([2594, height, width])
#
# print('Reading ISIC 2017')
# for idx in range(len(Tr_list)):
#     print(idx + 1)
#     img = imread(Tr_list[idx])
#     pil_img = Image.fromarray(img)
#     img = np.double(pil_img.resize((height, width), Image.BILINEAR))
#     Data_train_2017[idx, :, :, :] = img
#
#     b = Tr_list[idx]
#     a = b[0:len(Dataset_add)]
#     b = b[len(b) - 16: len(b) - 4]
#     add = (a + 'ISIC2018_Task1_Training_GroundTruth/' + b + '_segmentation.png')
#     img2 = imread(add)
#     pil_img2 = Image.fromarray(img2)
#     img2 = np.double(pil_img2.resize((height, width), Image.BILINEAR))
#     Label_train_2017[idx, :, :] = img2
#
# print('Reading ISIC 2018 finished')
#
#
# Train_img = Data_train_2017[0:2075, :, :, :]
# Test_img = Data_train_2017[2075:2594, :, :, :]
#
# Train_mask = Label_train_2017[0:2075, :, :]
# Test_mask = Label_train_2017[2075:2594, :, :]
#
# np.save('input/ISIC2018-2075-100-519/data_train', Train_img)
# np.save('input/ISIC2018-2075-100-519/data_test', Test_img)
#
# np.save('input/ISIC2018-2075-100-519/mask_train', Train_mask)
# np.save('input/ISIC2018-2075-100-519/mask_test', Test_mask)
# print(len(Train_img))
# print(len(Test_img))