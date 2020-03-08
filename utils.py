import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import os, random
import numpy as np

class ImageData:

    def __init__(self, load_size_h,load_size_w, channels, augment_flag,patch_h,patch_w,patch):
        self.load_size_h = load_size_h
        self.load_size_w = load_size_w
        self.patch_h= patch_h
        self.patch_w=patch_w
        self.patch = patch
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size_h, self.load_size_w])
        #print (img)
        if self.patch : 
            img = tf.image.random_crop(img,[self.patch_h,self.patch_w,3])
        #print (img)
        img = tf.cast(img, tf.float32) / 127.5 - 1
        

        if self.augment_flag :
            augment_size_h = self.load_size_h + (30 if self.load_size_h == 256 else 15)
            augment_size_w = self.load_size_w + (30 if self.load_size_w == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size_h,augment_size_w)

        return img

def load_test_data(image_path, size_h=256, size_w=256):
    
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(size_w, size_h))
    img = np.expand_dims(img, axis=0)
    #print (img)
    img = img/127.5 - 1
    #print (img.shape)
    return img

def augmentation(image, augment_size_h,augment_size_w):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size_h, augment_size_w])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    #print ((((images+1.)/2)*255.0).shape)
    return ((images+1.)/2)*255.0


def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image
    
    #print (img.shape)

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')
