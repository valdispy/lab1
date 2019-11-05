# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:54:59 2019

@author: valdis
"""

import os, glob
import pandas as pd
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt

def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature)
    
def content_loss(original_image, upscale_image): 
    original_image = tf.cast(original_image, tf.int64); upscale_image = tf.cast(upscale_image, tf.int64)
    content_loss = tf.reduce_sum(tf.square(tf.subtract(original_image, upscale_image)))
    return content_loss

if __name__ == "__main__":
    
    PSNR_image = 'PSNR.png'
    content_image = 'content_loss.png'
    
    work_dir = os.getcwd()
    tfrecords_folder = os.path.join(work_dir, 'tf_folder')
    possible_scales = (2,4,8,16,32)
    
    tfrerods_list = glob.glob(tfrecords_folder + '/*')
    tfdataset = tf.data.TFRecordDataset(tfrerods_list)
    
    feature = {'original_image': tf.io.FixedLenFeature([], tf.string), 
               'image_name': tf.io.FixedLenFeature([], tf.string)}
    for scale in possible_scales:
        feature.update({'scale_image_' + str(scale): tf.io.FixedLenFeature([], tf.string)})
    
    parsed_image_dataset = tfdataset.map(_parse_image_function)
    
    PSNR_tfdataset = []; content_tfdataset = []
    for image_features in parsed_image_dataset:
        
        print('Image =',image_features['image_name'].numpy().decode('utf-8'))
        original_image = tf.io.decode_jpeg(image_features['original_image']).numpy()
        
        PSNR_tfrecord = []; content_tfrecord = []; texture_tfrecord = []
        for scale_value in possible_scales: 
            upscale_image = tf.io.decode_jpeg(image_features['scale_image_' + str(scale_value)]).numpy()
            
            psnr_value = tf.image.psnr(original_image, upscale_image, max_val=255, name=None)     
            content_loss_value = content_loss(original_image, upscale_image)
            
            PSNR_tfrecord.append(psnr_value.numpy())
            content_tfrecord.append(content_loss_value.numpy())
            
        PSNR_tfdataset.append(PSNR_tfrecord)
        content_tfdataset.append(content_tfrecord)
        
    PSNR_frame = pd.DataFrame(PSNR_tfdataset, columns = [str(scale) for scale in possible_scales])
    content_frame = pd.DataFrame(content_tfdataset, columns = [str(scale) for scale in possible_scales])
    
    plt.figure()
    plt.ylabel('PSNR'); plt.xlabel('Scale')
    PSNR_boxplot = PSNR_frame.boxplot(grid = False)
    plt.savefig(os.path.join(work_dir, PSNR_image))
    
    plt.figure()
    plt.ylabel('Content loss'); plt.xlabel('Scale')
    PSNR_boxplot = content_frame.boxplot(grid = False)
    plt.savefig(os.path.join(work_dir, content_image))
