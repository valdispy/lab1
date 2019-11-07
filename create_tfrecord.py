# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:34:32 2019

@author: valdis
"""
import os, glob, cv2
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

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
        print(image_name)
        
        feature = {}
        for item in list_of_images:   
            image_path = os.path.join(folder, item)
            
            cv2_image = cv2.imread(image_path)
            height, width, _ = cv2_image.shape
            _, image_jpeg = cv2.imencode('.jpeg', cv2_image)
            bytes_image = image_jpeg.tobytes()

            if '_upscale_' not in item:
                image_shape = tf.image.decode_jpeg(bytes_image).shape
                feature.update({'original_image' : _bytes_feature(bytes_image), 
                                'image_name' : _bytes_feature(image_name.encode('utf-8')), 
                                'height' : _int64_feature(height), 'width' : _int64_feature(width)})
            else:
                scale = item.split('_upscale_')[-1].split('.')[0]
                feature.update({'scale_image_' + scale : _bytes_feature(bytes_image)})
        
        example_features = tf.train.Example(features = tf.train.Features(feature = feature))
    
        tfrecord_file = os.path.join(tfrecords_folder, image_name + '.tfrecord')
        writer = tf.io.TFRecordWriter(tfrecord_file)
        writer.write(example_features.SerializeToString())