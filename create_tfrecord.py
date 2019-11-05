# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:34:32 2019

@author: valdis
"""
import os, glob
import tensorflow as tf

def _int32_feature(value):
  return tf.train.Feature(int_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == "__main__":
    
    work_dir = os.getcwd()
    tfrecords_folder = os.path.join(work_dir, 'tf_folder')
    upscale_folder = os.path.join(work_dir, 'upscale_folder')
    
    sub_scale_folder = glob.glob(upscale_folder + '/*')
    for folder in sub_scale_folder: 
        
        image_name = folder.split('/')[-1]
        list_of_images = os.listdir(folder)
        
        feature = {}
        for item in list_of_images:   
            image_path = os.path.join(folder, item)
            bytes_image = open(image_path, 'rb').read()
            if '_upscale_' not in item:
                image_shape = tf.image.decode_jpeg(bytes_image).shape
                feature.update({'original_image' : _bytes_feature(bytes_image), 'image_name' : _bytes_feature(image_name.encode('utf-8'))})
            else:
                scale = item.split('_upscale_')[-1].split('.')[0]
                feature.update({'scale_image_' + scale : _bytes_feature(bytes_image)})
        
        example = tf.train.Example(features = tf.train.Features(feature = feature))
    
        tfrecord_file = os.path.join(tfrecords_folder, image_name + '.tfrecord')
        writer = tf.io.TFRecordWriter(tfrecord_file)
        writer.write(example.SerializeToString())