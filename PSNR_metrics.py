# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:54:49 2019

@author: valdis
"""
import cv2, os
from tqdm import tqdm
    
def get_image(image_path): 

    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)    
    height, width, _ =  original_image.shape  
    height_32 = height // 32; width_32 = width // 32
    if height != height_32 * 32 or width != width_32 * 32: 
        original_image = original_image[0:height_32 * 32, 0:width_32 * 32]
        cv2.imwrite(image_folder + current_image, original_image)

    return original_image

def image_sub_folders(scale_folder, upscale_folder, image_name, scale_value):
    
    scale_subfolder = os.path.join(scale_folder,image_name)
    upscale_subfolder = os.path.join(upscale_folder,image_name)
    
    if not os.path.exists(scale_subfolder):
        os.mkdir(scale_subfolder)    
    
    if not os.path.exists(upscale_subfolder):
        os.mkdir(upscale_subfolder)
    
    scale_path = os.path.join(scale_subfolder, image_name + '_scale_' + str(scale_value) + '.jpeg')
    upscale_path = os.path.join(upscale_subfolder, image_name + '_upscale_' + str(scale_value) + '.jpeg')
    
    return scale_path, upscale_path

if __name__ == "__main__":
    
    jpeg_quality = 80
    possible_scales = (2,4,8,16,32)
    image_folder = '/home/valdis/Desktop/Lab_1_sign/flickr_folder/'
    
    work_dir = os.getcwd()
    scale_folder = os.path.join(work_dir,'scale_folder')
    upscale_folder = os.path.join(work_dir,'upscale_folder')
    set_of_images = os.listdir(image_folder)
    
    if not os.path.exists(scale_folder):
        os.mkdir(scale_folder)

    if not os.path.exists(upscale_folder):
        os.mkdir(upscale_folder)    
    
    for current_image in tqdm(set_of_images):
        
        original_image = get_image(image_folder + current_image)
        
        for scale_value in possible_scales:
            
            image_name = os.path.splitext(current_image)[0]
            scale_path,  upscale_path = image_sub_folders(scale_folder, upscale_folder, image_name, scale_value)
            
            scale_image = cv2.resize(original_image, (0, 0), fx = 1./scale_value, fy = 1./scale_value)
            cv2.imwrite(scale_path, scale_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            
            upscale_image = cv2.imread(scale_path, cv2.IMREAD_UNCHANGED)
            upscale_image = cv2.resize(upscale_image, (0, 0), fx = scale_value, fy = scale_value, 
                                       interpolation = cv2.INTER_NEAREST)
            
            cv2.imwrite(upscale_path, upscale_image)                       

        
        cv2.imwrite(os.path.join(upscale_folder, image_name, image_name + '.jpeg'), original_image)        

    

