# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:11:27 2019

@author: valdis
"""

import os, glob
import tensorflow as tf

if __name__ == "__main__":

    possible_scales = (2,4,8,16,32)
    
    work_dir = os.getcwd()
    upscale_folder = os.path.join(work_dir, 'upscale_folder')
    tf_folder = os.path.join(work_dir, 'tf_folder')
    
    if not os.path.exists(tf_folder):
        os.mkdir(tf_folder)
    
    for scale_value in possible_scales:
        
        list_of_scale = glob.glob(upscale_folder + '/*/*_upscale_' + str(scale_value) + '.jpeg')
        tf_dataset = tf.data.Dataset.from_tensor_slices(list_of_scale).map(tf.io.read_file)

        tf_path = os.path.join(tf_folder,'record_' + str(scale_value) + '.tfrecord')
        tf_record = tf.data.experimental.TFRecordWriter(tf_path)
        tf_record.write(tf_dataset)