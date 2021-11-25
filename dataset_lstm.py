# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:59:43 2021

@author: gyuha
"""

import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import cv2
import random
from glob import glob

class LSTM_Dataset(Dataset):
    def __init__(self, cl=None, dicom_paths=[], transform=None, max_num=6):

        self.dicom_paths = dicom_paths
        self.transform = transform
        self.cl = cl
        self.max_num = max_num
        
    def __getitem__(self, index):
        
        cl = self.cl
        dicom_paths = self.dicom_paths[index]
        max_num = self.max_num
        seg_num = int(dicom_paths[0].split('_')[-1][:-4])
        
        # 1 stage
        # # M1
        # if seg_num==1 or seg_num==8 or seg_num==15 or seg_num==22 or seg_num==29:           
        #     max_resample = 9
        # # M2
        # elif seg_num==2 or seg_num==9 or seg_num==16 or seg_num==23 or seg_num==30:           
        #     max_resample = 9
        # # M3
        # elif seg_num==3 or seg_num==10 or seg_num==17 or seg_num==24 or seg_num==31:           
        #     max_resample = 9
        # # M4
        # elif seg_num==34 or seg_num==37 or seg_num==40 or seg_num==43:           
        #     max_resample = 6
        # # M5
        # elif seg_num==35 or seg_num==38 or seg_num==41 or seg_num==44:           
        #     max_resample = 6
        # # M6
        # elif seg_num==36 or seg_num==39 or seg_num==42 or seg_num==45:           
        #     max_resample = 6
        # # I
        # elif seg_num==4 or seg_num==11 or seg_num==18 or seg_num==25:           
        #     max_resample = 7
        # # L
        # elif seg_num==5 or seg_num==12 or seg_num==19 or seg_num==26:           
        #     max_resample = 7
        # # C
        # elif seg_num==7 or seg_num==14 or seg_num==21 or seg_num==28 or seg_num==32 or seg_num==33:           
        #     max_resample = 10
        # # IC
        # else:           
        #     max_resample = 7
                
        # 2 stage
        #M1
        if seg_num==1 or seg_num==8 or seg_num==15 or seg_num==22 or seg_num==29:           
            max_resample = 5
        # M2
        elif seg_num==2 or seg_num==9 or seg_num==16 or seg_num==23 or seg_num==30:           
            max_resample = 5
        # M3
        elif seg_num==3 or seg_num==10 or seg_num==17 or seg_num==24 or seg_num==31:           
            max_resample = 5
        # M4
        elif seg_num==34 or seg_num==37 or seg_num==40 or seg_num==43:           
            max_resample = 4
        # M5
        elif seg_num==35 or seg_num==38 or seg_num==41 or seg_num==44:           
            max_resample = 4
        # M6
        elif seg_num==36 or seg_num==39 or seg_num==42 or seg_num==45:           
            max_resample = 4
        # I
        elif seg_num==4 or seg_num==11 or seg_num==18 or seg_num==25:           
            max_resample = 5
        # L
        elif seg_num==5 or seg_num==12 or seg_num==19 or seg_num==26:           
            max_resample = 5
        # C
        elif seg_num==7 or seg_num==14 or seg_num==21 or seg_num==28 or seg_num==32 or seg_num==33:           
            max_resample = 6
        # IC
        else:           
            max_resample = 5       
 
        # stack images
        images = []
        
        for dicom_path in dicom_paths:
            dataset= pydicom.read_file(dicom_path, force=True)
            dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

            image = dataset.pixel_array        
            images.append(image)
        
        # resample segment
        sample_num = np.random.randint(0,max_resample-len(images)+1)
        
        if sample_num > len(images):
            sample_num = len(images)
    
        index_list = random.sample([i for i in range(len(images))], sample_num)
        
        u_images = []
        for i in range(len(images)):
            u_images.append(images[i])
            
            if i in index_list:
                u_images.append(images[i])    
        
        # set label
        if cl == 'eic' or cl == 'oi':
            label = 1
        else: label = 0
        
        # transform
        if self.transform:
            u_images = self.transform(u_images)
        
        # zero padding
        n, c, h, w = u_images.shape
        
        if n < max_num:
            padding = torch.zeros((max_num-n,c,h,w)) # (max_num-n, c, h, w)
            u_images = torch.cat((padding,u_images),dim=0) # (max_num, c, h, w)      
        
        sample = {'img': u_images, 'annot': torch.Tensor([label])}
            
        return sample
    
    def __len__(self):
        return len(self.dicom_paths)

class LSTM_Gil_Normal_Dataset(Dataset):
    def __init__(self, cl=None, dicom_paths=[], transform=None, max_num=6):

        self.dicom_paths = dicom_paths
        self.transform = transform
        self.cl = cl
        self.max_num = max_num
        
    def __getitem__(self, index):
        
        cl = self.cl
        dicom_paths = self.dicom_paths[index]
        max_num = self.max_num
        seg_num = int(dicom_paths[0].split('_')[-1][:-4])
        
        
        dataset = pydicom.read_file(glob('D:/G_train_region_DECT_Norm/' + dicom_paths[0].split('/')[2] + '/NCCT/*.dcm')[0])
        thickness = float(dataset[('0018', '0050')].value)
        
        if 4.0 <= thickness <= 5.0:
            #M1
            if seg_num==1 or seg_num==8 or seg_num==15 or seg_num==22 or seg_num==29:           
                max_resample = 5
            # M2
            elif seg_num==2 or seg_num==9 or seg_num==16 or seg_num==23 or seg_num==30:           
                max_resample = 5
            # M3
            elif seg_num==3 or seg_num==10 or seg_num==17 or seg_num==24 or seg_num==31:           
                max_resample = 5
            # M4
            elif seg_num==34 or seg_num==37 or seg_num==40 or seg_num==43:           
                max_resample = 4
            # M5
            elif seg_num==35 or seg_num==38 or seg_num==41 or seg_num==44:           
                max_resample = 4
            # M6
            elif seg_num==36 or seg_num==39 or seg_num==42 or seg_num==45:           
                max_resample = 4
            # I
            elif seg_num==4 or seg_num==11 or seg_num==18 or seg_num==25:           
                max_resample = 5
            # L
            elif seg_num==5 or seg_num==12 or seg_num==19 or seg_num==26:           
                max_resample = 5
            # C
            elif seg_num==7 or seg_num==14 or seg_num==21 or seg_num==28 or seg_num==32 or seg_num==33:           
                max_resample = 6
            # IC
            else:           
                max_resample = 5       
        
        elif 3.0 <= thickness < 4.0:
            #M1
            if seg_num==1 or seg_num==8 or seg_num==15 or seg_num==22 or seg_num==29 or seg_num==36 or seg_num==43 or seg_num==50 or seg_num==54:        
                max_resample = 5
            # M2
            elif seg_num==2 or seg_num==9 or seg_num==16 or seg_num==23 or seg_num==30 or seg_num==37 or seg_num==44 or seg_num==51 or seg_num==55:          
                max_resample = 5
            # M3
            elif seg_num==3 or seg_num==10 or seg_num==17 or seg_num==24 or seg_num==31 or seg_num==38 or seg_num==45 or seg_num==52 or seg_num==56:          
                max_resample = 5
            # M4
            elif seg_num==59 or seg_num==62 or seg_num==65 or seg_num==68 or seg_num==71 or seg_num==74:          
                max_resample = 4
            # M5
            elif seg_num==60 or seg_num==63 or seg_num==66 or seg_num==69 or seg_num==72 or seg_num==75:          
                max_resample = 4
            # M6
            elif seg_num==61 or seg_num==64 or seg_num==67 or seg_num==70 or seg_num==73 or seg_num==76:           
                max_resample = 4
            # I
            elif seg_num==4 or seg_num==11 or seg_num==18 or seg_num==25 or seg_num==32 or seg_num==39 or seg_num==46:           
                max_resample = 5
            # L
            elif seg_num==5 or seg_num==12 or seg_num==19 or seg_num==26 or seg_num==33 or seg_num==40 or seg_num==47:           
                max_resample = 5
            # C
            elif seg_num==7 or seg_num==14 or seg_num==21 or seg_num==28 or seg_num==35 or seg_num==42 or seg_num==49 or seg_num==53 or seg_num==57 or seg_num==58:          
                max_resample = 6
            # IC
            else:           
                max_resample = 5 
            
        # stack images
        images = []
        
        for dicom_path in dicom_paths:
            dataset= pydicom.read_file(dicom_path, force=True)
            dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

            image = dataset.pixel_array        
            images.append(image)
        
        # resample segment
        sample_num = np.random.randint(0,max_resample-len(images)+1)
        
        if sample_num > len(images):
            sample_num = len(images)
    
        index_list = random.sample([i for i in range(len(images))], sample_num)
        
        u_images = []
        for i in range(len(images)):
            u_images.append(images[i])
            
            if i in index_list:
                u_images.append(images[i])    
        
        # set label
        if cl == 'eic' or cl == 'oi':
            label = 1
        else: label = 0
        
        # transform
        if self.transform:
            u_images = self.transform(u_images)
        
        # zero padding
        n, c, h, w = u_images.shape
        
        if n < max_num:
            padding = torch.zeros((max_num-n,c,h,w)) # (max_num-n, c, h, w)
            u_images = torch.cat((padding,u_images),dim=0) # (max_num, c, h, w)      
        
        sample = {'img': u_images, 'annot': torch.Tensor([label])}
            
        return sample
    
    def __len__(self):
        return len(self.dicom_paths)

class LSTM_Test_Dataset(Dataset):
    def __init__(self, cl=None, dicom_paths=[], transform=None, max_num=6):

        self.dicom_paths = dicom_paths
        self.transform = transform
        self.cl = cl
        self.max_num = max_num
        
    def __getitem__(self, index):
        
        cl = self.cl
        dicom_paths = self.dicom_paths[index]
        max_num = self.max_num
      
        images = []
        
        for dicom_path in dicom_paths:
            dataset= pydicom.read_file(dicom_path, force=True)
            dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            
            image = dataset.pixel_array
            images.append(image)
            
        if cl == 'eic' or cl == 'oi':
            label = 1
        else: label = 0
               
        if self.transform:
            images = self.transform(images)
        
        n, c, h, w = images.shape
        
        if n < max_num:
            padding = torch.zeros((max_num-n,c,h,w)) # (max_num-n, c, h, w)
            images = torch.cat((padding,images),dim=0) # (max_num, c, h, w)      
        
        sample = {'img': images, 'annot': torch.Tensor([label])}
            
        return sample
    
    def __len__(self):
        return len(self.dicom_paths)

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]

    imgs = torch.stack(imgs, axis=0)  
    annots = torch.stack(annots, axis=0)  

    return {'img': imgs, 'annot': annots}

# class RandomUpsample(object):
    
#     def __call__(self, images):
        
#         sample_num = min(np.random.randint(0,10-len(images)),len(images))
        
#         if sample_num > len(images):
#             sample_num = len(images)
        
#         index_list = random.sample([i for i in range(len(images))], sample_num)
        
#         u_images = []
#         for i in range(len(images)):
#             u_images.append(images[i])
            
#             # Upsample
#             if i in index_list:
#                 u_images.append(images[i])
        
#         return u_images

class Crop(object):

    def __call__(self, images):
        
        crop_images = []
        for image in images:
            mask = np.where(image>image.min(),1,0)
            # Bounding box.
            horizontal_indicies = np.where(np.any(mask, axis=0))[0]
            vertical_indicies = np.where(np.any(mask, axis=1))[0]
            if horizontal_indicies.shape[0]:
                xl, xr = horizontal_indicies[[0, -1]]
                yl, yr = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                xr += 1
                yr += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                xl, xr, yl, yr = 0, 0, 0, 0
                
            crop_images.append(image[yl:yr,xl:xr])
        
        return crop_images
    
class Resize(object):
    
    def __init__(self, img_size=[362,362]):
        self.img_size = img_size

    def __call__(self, images):
        
        resize_images = []
        for image in images:
            resize_image = cv2.resize(image, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            
            resize_images.append(resize_image)
            
        return resize_images

class RandomResize(object):
    
    def __init__(self, img_size=[256,362]):
        self.img_size = img_size

    def __call__(self, images):
        resize = np.random.randint(self.img_size[0], self.img_size[1])
        
        resize_images = []
        for image in images:
            resize_image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_LINEAR)
            
            resize_images.append(resize_image)
            
        return resize_images

class Zoom(object):

    def __init__(self, zoom=(0.6, 1.6)):
        self.zoom = zoom

    def __call__(self, images):
        zoom = np.random.randint(self.zoom[0] * 10, (self.zoom[1]) * 10) / 10
        
        zoom_images = []
        
        for image in images:
            height, width = image.shape
            r_height, r_width = int(height * zoom), int(width * zoom)
            resize_image = cv2.resize(image.astype(np.float32), (r_width, r_height), interpolation=cv2.INTER_LINEAR)
            zoom_image = np.full((height, width),
                                 image.min(), dtype=np.float64)
    
            margin_height = abs(int((r_height - height) / 2))
            margin_width = abs(int((r_width - width) / 2))
            
            if zoom < 1:
                zoom_image[margin_height:margin_height + r_height, margin_width:margin_width + r_width] = resize_image
            
            else:
                zoom_image = resize_image[margin_height:margin_height + height, margin_width:margin_width + width]
            
            zoom_images.append(zoom_image.astype(np.float32))
            
        return zoom_images

class Padding(object):

    def __init__(self, img_size=[512,512]):
        self.img_size = img_size

    def __call__(self, images):
        padding_images = []
        for image in images:
            height, width = image.shape
            p_height, p_width = self.img_size[1], self.img_size[0]
            padding_image = np.zeros((p_height,p_width))
            
            padding_image[int((p_height-height)/2):int((p_height-height)/2)+height,int((p_width-width)/2):int((p_width-width)/2)+width] = image
            
            padding_images.append(padding_image)
            
        return padding_images

class Windowing(object):

    def __init__(self, windowing, WC=40, WW=80, rescale=True):
        self.windowing = windowing
        self.WC = WC
        self.WW = WW
        self.rescale = rescale
    def __call__(self, images):
        w_images = []
        
        for image in images:
            height, width = image.shape
            w_image = self.windowing(image, self.WC, self.WW, self.rescale)
            
            if len(w_image.shape)==2:
                w_image = np.stack((w_image,)*3, axis=-1)
                
            w_images.append(w_image)
                
        return w_images

class Flip(object):

    def __call__(self, images):
        f_images = []
        
        for image in images:
            height, width = image.shape
            f_image = image[:, ::-1]
            
            f_images.append(f_image.astype(np.float32))
            
        return f_images

class Rotate(object):

    def __init__(self, angle=10):
        self.angle = angle
        
    def __call__(self, images):
        angle = np.random.randint(-self.angle, self.angle)
        
        rot_images = []
        
        for image in images:
            height, width = image.shape
            matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            rot_image = cv2.warpAffine(image.astype(np.float32), matrix, (width, height))
            
            rot_images.append(rot_image.astype(np.float32))
            
        return rot_images

class Shear(object):

    def __init__(self, sx_range=[0,0.1], sy_range=[0,0.1]):
        self.sx_range = sx_range
        self.sy_range = sy_range
        
    def __call__(self, images):
        sx = np.random.randint(self.sx_range[0]*10, self.sx_range[1]*10)/10
        sy = np.random.randint(self.sy_range[0]*10, self.sy_range[1]*10)/10
        
        s_images = []
        
        for image in images:
            height, width = image.shape
            matrix = np.array([[1,sx,0],[sy,1,0]])
            s_image = cv2.warpAffine(image.astype(np.float32), matrix, (width, height))
            
            s_images.append(s_image.astype(np.float32))
            
        return s_images

class Elastic(object):
    
    def __init__(self, alpha=15, sigma=0.25, grid_scale=4):
        self.alpha = alpha
        self.sigma = sigma
        self.grid_scale = grid_scale
    def __call__(self, images):
        random_state = np.random.RandomState(None)
        
        shape_size = images[0].shape
        alpha = shape_size[0]*random.random()*self.alpha
        sigma = shape_size[0]*self.sigma
        
        # Downscaling the random grid and then upsizing post filter
        # improves performance. Approx 3x for scale of 4, diminishing returns after.
        grid_scale = self.grid_scale
        alpha //= grid_scale  # Does scaling these make sense? seems to provide
        sigma //= grid_scale  # more similar end result when scaling grid used.
        grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)
    
        blur_size = int(4 * sigma) | 1
        rand_x = cv2.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        rand_y = cv2.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        if grid_scale > 1:
            rand_x = cv2.resize(rand_x, shape_size[::-1])
            rand_y = cv2.resize(rand_y, shape_size[::-1])
               
        grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        grid_x = (grid_x + rand_x).astype(np.float32)
        grid_y = (grid_y + rand_y).astype(np.float32)
        
        distorted_images = []
        
        for image in images:
            distorted_image = cv2.remap(image, grid_x, grid_y,
                borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)
            
            distorted_images.append(distorted_image.astype(np.float32))
            
        return distorted_images

class Brightness(object):
    
    def __init__(self, brightness=(-15,15)):
        self.brightness = brightness
        
    def __call__(self, images):
        brightness = np.random.randint(self.brightness[0], self.brightness[1])
        
        b_images = []
        
        for image in images:
            mask = np.where(image>image.min(),0,1)
            b_image = image + brightness
            b_image = np.where(b_image>0,b_image,0)
            b_image = np.where(mask==1,0,b_image)
        
            b_images.append(b_image)
        
        return b_images
    
class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, images):
        norm_images = []
        
        for image in images:
            norm_image = ((image.astype(np.float32) - self.mean) / self.std)
            
            norm_images.append(torch.from_numpy(norm_image).to(torch.float32).permute(2,0,1))
            
        return torch.stack(norm_images, axis=0)

class Identity(object):
    def __call__(self, images):           
        return images

def augment_list():
    l = [
        Zoom(),
        Rotate(),
        Shear(),
        Elastic(),
        Brightness(),
        Flip(),
        Identity(),
    ]
    
    return l
      
class RandAugment:
    def __init__(self, n=2):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        random.shuffle(ops)
        for op in ops:
            img = op(img)

        return img

# class RandAugment:
#     def __init__(self, krange=[0,4]):
#         self.krange = krange
#         self.augment_list = augment_list()

#     def __call__(self, img):
#         k = np.random.randint(self.krange[0], self.krange[1])
        
#         if k==0:
#             return img
#         else:
#             ops1 = random.sample(self.augment_list, k)     
#             random.shuffle(ops1)
#             for op in ops1:
#                 img = op(img)
        
#             return img