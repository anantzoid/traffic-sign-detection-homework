from __future__ import print_function
import zipfile
import os
from skimage import io, color, exposure, transform
import torchvision.transforms as transforms
from keras.preprocessing.image import *
import numpy as np
# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

to_numpy = lambda im: np.array(im)
transform_random_shift = lambda im: random_shift(im, 0.1, 0.1)
transform_random_rotation = lambda im: random_rotation(im, 10.0)
transform_random_shear = lambda im: random_shear(im, 0.1)
transform_random_zoom = lambda im: random_zoom(im, 0.2)
IMG_SIZE = 32#48

def preprocess_img(img):
    # Histogram normalization in y    
    img = np.array(img)    
    #print(img.shape)
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))              
    #img = np.rollaxis(img,-1) 
    #print(img.shape)
    
    return img


data_transforms = transforms.Compose([    
    transforms.Lambda(preprocess_img),
    transforms.Lambda(transform_random_shift),
    transforms.Lambda(transform_random_rotation),
    transforms.Lambda(transform_random_zoom),
    transforms.Lambda(transform_random_shear),    
    transforms.ToTensor()
    
])
#transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
val_data_transforms = transforms.Compose([
    transforms.Lambda(preprocess_img),
    transforms.ToTensor()
    ])

def initialize_data(folder):
    train_zip = folder + '/train_images.zip'
    test_zip = folder + '/test_images.zip'
    if not os.path.exists(train_zip) or not os.path.exists(test_zip):
        raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
              + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2017/data '))
    # extract train_data.zip to train_data
    train_folder = folder + '/train_images'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found, extracting ' + train_zip)
        zip_ref = zipfile.ZipFile(train_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
    # extract test_data.zip to test_data
    test_folder = folder + '/test_images'
    if not os.path.isdir(test_folder):
        print(test_folder + ' not found, extracting ' + test_zip)
        zip_ref = zipfile.ZipFile(test_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()

    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)
                for f in os.listdir(train_folder + '/' + dirs):
                    if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                        # move file to validation folder
                        os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
