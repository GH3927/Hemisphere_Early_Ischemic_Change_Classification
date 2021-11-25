# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:02:57 2021

@author: gyuha
"""

import os
import numpy as np
import pandas as pd
import pydicom
from dcm import get_windowing, windowing
import torch
from dataset import Resizer, Normalizer, Windowing
from torchvision import transforms
from glob import glob
from efficientnet.model import EfficientNet_f

# Hyperparameters
opt = {
    'project':'w4080',
    'compound_coef':4,
    'input_size':[512,256],
    'data_path':"D:/AJ_Eval_DATA_hemi_v3/",
    'save_path':"Z:/Stroke/SharingFolder/cELVO/manual_dataset_dect_norm_features_v5/",
    'WC':40,
    'WW':80,
    'Windowing':windowing,
    'mean':[10.604362079468231, 10.604362079468231, 10.604362079468231],
    'std':[15.79586109964471, 15.79586109964471, 15.79586109964471],
}

# Load weight paths
eic_weight_paths = './weights/eic_w4080_dect_norm+gil.pth'

# Cuda
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
Tensor = torch.cuda.FloatTensor if use_cuda else torch.Tensor
if use_cuda:
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)

# Transform
transform = transforms.Compose([
                                Resizer(opt['input_size']),
                                Windowing(windowing=opt['Windowing'],WC=opt['WC'],WW=opt['WW'],rescale=False),
                                Normalizer(mean=opt['mean'], std=opt['std'])
                               ])

# Model
model = EfficientNet_f.from_pretrained('efficientnet-b{}'.format(opt['compound_coef']))
model._fc = torch.nn.Linear(1792, 1)
model.to(device)
model.eval()

# Load dataset
train_csv = opt['data_path']+'df_train.csv'
test_csv = opt['data_path']+'df_test.csv'

train_annot = pd.read_csv(train_csv)
id = train_annot['ID']
id = id.drop_duplicates()
train_HR = id.to_list()

test_annot = pd.read_csv(test_csv)
id = test_annot['ID']
id = id.drop_duplicates()
test_HR = id.to_list()

train_paths = []
test_paths = []

for HR in train_HR:
    dcm_paths = glob(opt['data_path']+'dataset_dect_norm/{}/*.dcm'.format(str(HR).rjust(3,'0')))
    for dcm_path in dcm_paths:
        train_paths.append(dcm_path)
    
for HR in test_HR:
    dcm_paths = glob(opt['data_path']+'dataset_dect_norm/{}/*.dcm'.format(str(HR).rjust(3,'0')))
    for dcm_path in dcm_paths:
        test_paths.append(dcm_path)

# save features
os.makedirs(opt['save_path']+'train/eic',exist_ok=True)
os.makedirs(opt['save_path']+'test/eic',exist_ok=True)

for i in range(len(train_paths)):
    dataset = pydicom.read_file(train_paths[i])
    img = dataset.pixel_array
    
    # rescale
    _, _, intercept, slope = get_windowing(dataset)
    img = (img * slope + intercept)
    
    height, width = img.shape

    img_r = img[:,:int(width/2)-int(width*0.02)]
    img_l = img[:,int(width/2)+int(width*0.02):]
    
    t_img_r = transform(img_r).unsqueeze(0).permute(0,3,1,2)
    t_img_l = transform(img_l).unsqueeze(0).permute(0,3,1,2)
    
    # save eic feature
    model.load_state_dict(torch.load(eic_weight_paths))
    feature_r = model(t_img_r.cuda())
    feature_l = model(t_img_l.cuda())

    np.save(opt['save_path']+'train/eic/{}_R'.format(train_paths[i].split('\\')[-1][:-4]), feature_r.cpu().detach().numpy())
    np.save(opt['save_path']+'train/eic/{}_L'.format(train_paths[i].split('\\')[-1][:-4]), feature_l.cpu().detach().numpy())

for i in range(len(test_paths)):
    dataset = pydicom.read_file(test_paths[i])
    img = dataset.pixel_array
    
    # rescale
    _, _, intercept, slope = get_windowing(dataset)
    img = (img * slope + intercept)
    
    height, width = img.shape

    img_r = img[:,:int(width/2)-int(width*0.02)]
    img_l = img[:,int(width/2)+int(width*0.02):]
    
    t_img_r = transform(img_r).unsqueeze(0).permute(0,3,1,2)
    t_img_l = transform(img_l).unsqueeze(0).permute(0,3,1,2)
    
    # save eic feature
    model.load_state_dict(torch.load(eic_weight_paths))
    feature_r = model(t_img_r.cuda())
    feature_l = model(t_img_l.cuda())
    
    np.save(opt['save_path']+'test/eic/{}_R'.format(test_paths[i].split('\\')[-1][:-4]), feature_r.cpu().detach().numpy())
    np.save(opt['save_path']+'test/eic/{}_L'.format(test_paths[i].split('\\')[-1][:-4]), feature_l.cpu().detach().numpy())