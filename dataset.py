import torch
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
import numpy as np
import cv2
import random
import pydicom
from dcm import get_windowing

class EICDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_path = image_dir
        self.transform = transform
        
    def __getitem__(self, index):
        # Load Image
        image_fn = self.image_path[index]
        dataset = pydicom.read_file(image_fn, force=True)
        dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        
        image = dataset.pixel_array.astype(np.float32)
    
        # windowing 파라미터 획득
        _, _, intercept, slope = get_windowing(dataset)
        
        # rescale 수행
        image = (image * slope + intercept)
        
        label = 0
        
        if image_fn.split('/')[-1][0] == 'e':
            label = 1
        else: label = 0

        if self.transform:
            image = self.transform(image)
        sample = {'img': image, 'annot': torch.Tensor([label])}
            
        return sample
    
    def __len__(self):
        return len(self.image_path)

class OIDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_path = image_dir
        self.transform = transform
        
    def __getitem__(self, index):
        # Load Image
        image_fn = self.image_path[index]
        dataset = pydicom.read_file(image_fn, force=True)
        dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        
        image = dataset.pixel_array.astype(np.float32)
           
        # windowing 파라미터 획득
        _, _, intercept, slope = get_windowing(dataset)
        
        # rescale 수행
        image = (image * slope + intercept)
        
        label = 0
        
        if image_fn.split('/')[-1][0] == 'o' or image_fn.split('/')[-1][:4] == 'eic+':
            label = 1
        else: label = 0

        if self.transform:
            image = self.transform(image)
        sample = {'img': image, 'annot': torch.Tensor([label])}
            
        return sample
    
    def __len__(self):
        return len(self.image_path)

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    imgs = imgs.permute(0, 3, 1, 2)

    annots = torch.from_numpy(np.stack(annots, axis=0))    

    return {'img': imgs, 'annot': annots}

class Crop(object):

    def __call__(self, image):
        mask = np.where(image>0,1,0)
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
            
        crop_image = image[yl:yr,xl:xr]
        
        return crop_image.astype(np.float32)

class Windowing(object):

    def __init__(self, windowing, WC=40, WW=80, rescale=True):
        self.windowing = windowing
        self.WC = WC
        self.WW = WW
        self.rescale = rescale
    def __call__(self, image):
        height, width = image.shape
        w_image = self.windowing(image, self.WC, self.WW, self.rescale)
        
        if len(w_image.shape)==2:
            w_image = np.stack((w_image,)*3, axis=-1)
        
        return w_image

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=[512,256]):
        self.img_size = img_size

    def __call__(self, image):
        height, width = image.shape

        resize_image = cv2.resize(image.astype(np.float32), (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        
        return resize_image.astype(np.float32)

class Flip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        flip_image = image[:, ::-1]
        return flip_image.astype(np.float32)

class Rotate(object):

    def __init__(self, angle=10):
        self.angle = angle

    def __call__(self, image):
        angle = np.random.randint(-self.angle, self.angle)

        height, width = image.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rot_image = cv2.warpAffine(image.astype(np.float32), matrix, (width, height))

        return rot_image.astype(np.float32)

class Zoom(object):

    def __init__(self, zoom=(0.6, 1.6)):
        self.zoom = zoom

    def __call__(self, image):
        zoom = np.random.randint(self.zoom[0] * 10, (self.zoom[1]) * 10) / 10

        height, width = image.shape
        r_height, r_width = int(height * zoom), int(width * zoom)
        resize_image = cv2.resize(image.astype(np.float32), (r_width, r_height), interpolation=cv2.INTER_LINEAR)
        zoom_image = np.full((height, width),
                             image.min(), dtype=np.float64)

        margin_height = abs(int((r_height - height) / 2))
        margin_width = abs(int((r_width - width) / 2))

        # Right
        if image[:, 0].mean() < image[:, -1].mean():
            if zoom <= 1:
                zoom_image[margin_height:margin_height + r_height, width - r_width:] = resize_image
            else:
                zoom_image = resize_image[margin_height:margin_height + height, r_width - width:]

        # Left
        else:
            if zoom <= 1:
                zoom_image[margin_height:margin_height + r_height, :r_width:] = resize_image
            else:
                zoom_image = resize_image[margin_height:margin_height + height, :width]

        return zoom_image.astype(np.float32)

class Elastic(object):
    
    def __init__(self, alpha=10, sigma=0.15, grid_scale=4):
        self.alpha = alpha
        self.sigma = sigma
        self.grid_scale = grid_scale
        
    def __call__(self, image):
        random_state = np.random.RandomState(None)
        
        shape_size = image.shape
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
    
        distorted_image = cv2.remap(image.astype(np.float32), grid_x, grid_y,
            borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)

        return distorted_image.astype(np.float32)

class Brightness(object):
    def __init__(self, brightness=(-15,15)):
        self.brightness = brightness
        
    def __call__(self, image):
        brightness = np.random.randint(self.brightness[0], self.brightness[1])

        mask = np.where(image > 0, 0, 1)
        b_image = image + brightness
        b_image = np.where(b_image > 0, b_image, 0)
        b_image = np.where(mask == 1, 0, b_image)

        return b_image.astype(np.float32)

class Shear(object):

    def __init__(self, sx_range=[0,0.1], sy_range=[0,0.1]):
        self.sx_range = sx_range
        self.sy_range = sy_range
        
    def __call__(self, image):
        sx = np.random.randint(self.sx_range[0]*10, self.sx_range[1]*10)/10
        sy = np.random.randint(self.sy_range[0]*10, self.sy_range[1]*10)/10
        
        height, width = image.shape
        matrix = np.array([[1,sx,0],[sy,1,0]])
        s_image = cv2.warpAffine(image.astype(np.float32), matrix, (width, height))
        
        return s_image.astype(np.float32)

class Identity(object):
    def __call__(self, image):           
        return image

class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, image):
        norm_image = ((image.astype(np.float32) - self.mean) / self.std)
        
        return torch.from_numpy(norm_image).to(torch.float32)

def augment_list():
    l = [
        Zoom(),
        Elastic(),
        Brightness(),
        Flip(),
        Rotate(),
        Shear(),
        Identity(),
        iaa.SomeOf(1,[
               iaa.GaussianBlur(sigma=(1.0, 2.0)),
               iaa.MotionBlur(k=(3, 7)),
               iaa.AverageBlur(k=(1, 5))
           ]).augment_image
    ]

    return l

class RandAugment:
    def __init__(self, n=2):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img)

        return img