# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:52:49 2019

@author: valdis
"""
import os, glob
import pandas as pd
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt 

def _parse_image_function(example_proto):
    features = create_fetures(possible_scales)
    return tf.io.parse_single_example(example_proto, features)

def create_fetures(possible_scales):
    features = {'original_image': tf.io.FixedLenFeature([], tf.string), 
                'image_name': tf.io.FixedLenFeature([], tf.string),
                'height': tf.io.FixedLenFeature([], tf.int64),
                'width': tf.io.FixedLenFeature([], tf.int64)}
    for scale in possible_scales:
        features.update({'scale_image_' + scale: tf.io.FixedLenFeature([], tf.string)})
    return features
    
def loss_functions(original_image, upscale_image, total_length): 
    
    psnr_value = tf.image.psnr(original_image, upscale_image, max_val=255).numpy()
    content_loss = tf.reduce_sum(tf.square(tf.subtract(original_image, upscale_image))).numpy()   
    
    original_gram_matrix = tf.reduce_sum(tf.matmul(original_image, original_image, transpose_b=True))/total_length
    upscale_gram_matrix = tf.reduce_sum(tf.matmul(upscale_image,upscale_image,transpose_b=True))/total_length
    texture_loss = tf.keras.backend.mean(tf.square(original_gram_matrix-upscale_gram_matrix)).numpy()
    
    return psnr_value, content_loss/total_length, texture_loss/total_length
   
if __name__ == "__main__":
    
    PSNR_image = 'PSNR.png'
    content_image = 'content_loss.png'
    texture_image = 'texture_loss.png'
    
    work_dir = os.getcwd()
    tfrecords_folder = os.path.join(work_dir, 'tf_folder')
    possible_scales = [str(scale) for scale in [2,4,8,16,32]]
    tfrerods_list = glob.glob(tfrecords_folder + '/*')
    tfdataset = tf.data.TFRecordDataset(tfrerods_list)
    
    parsed_image_dataset = tfdataset.map(_parse_image_function)
    
    PSNR_tfdataset = []; content_tfdataset = []; texture_tfdataset = []
    for image_features in parsed_image_dataset:
        
        print(image_features['image_name'].numpy().decode('utf-8'))
        original_image = tf.io.decode_jpeg(image_features['original_image'])
        original_image = tf.cast(original_image, tf.int64).numpy()
    
        height = image_features['height'].numpy(); width = image_features['width'].numpy() 
        total_length = 3 * height * width
        
        PSNR_tfrecord = []; content_tfrecord = []; texture_tfrecord = []
        for scale_value in possible_scales:
            upscale_image = tf.io.decode_jpeg(image_features['scale_image_' + str(scale_value)])
            upscale_image = tf.cast(upscale_image, tf.int64).numpy()
            
            psnr_value, content_loss, texture_loss = loss_functions(original_image, upscale_image, total_length)
        
            PSNR_tfrecord.append(psnr_value); content_tfrecord.append(content_loss)
            texture_tfrecord.append(texture_loss)
           
        PSNR_tfdataset.append(PSNR_tfrecord); content_tfdataset.append(content_tfrecord)
        texture_tfdataset.append(texture_tfrecord)
        
    PSNR_frame = pd.DataFrame(PSNR_tfdataset, columns = possible_scales)
    content_frame = pd.DataFrame(content_tfdataset, columns = possible_scales)
    texture_frame = pd.DataFrame(texture_tfdataset, columns = possible_scales)
    
    plt.figure()
    plt.ylabel('PSNR'); plt.xlabel('Scale')
    PSNR_boxplot = PSNR_frame.boxplot(grid = False)
    plt.savefig(os.path.join(work_dir, PSNR_image))
    
    plt.figure()
    plt.ylabel('Content loss'); plt.xlabel('Scale')
    content_boxplot = content_frame.boxplot(grid = False)
    plt.savefig(os.path.join(work_dir, content_image))
    
    plt.figure()
    plt.ylabel('Texture loss'); plt.xlabel('Scale')
    texture_boxplot = texture_frame.boxplot(grid = False)
    plt.savefig(os.path.join(work_dir, texture_image))
