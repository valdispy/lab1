# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 08:49:36 2019

@author: valdis
"""
import urllib, os
from flickrapi import FlickrAPI

def get_from_flickr(flickr, keyward, image_count, flickr_folder):
    
    photos_list = flickr.walk(text = keyward,
                 tag_mode = 'all', tags = keyward,
                 extras = 'url_c', per_page = image_count)
    
    index = 0  
    for photo in photos_list:
        if index < image_count:
            print("Image index - {}".format(index))
            try:
                url = photo.get('url_c')
                urllib.request.urlretrieve(url, flickr_folder + os.path.basename(url))
                index += 1
            except:
                print("Broken {} image link".format(index))
        else:
            break

if __name__ == "__main__":
    
    image_count = 10; keyward = 'dog'
    flickr_folder = '/home/valdis/Desktop/Lab_1/flickr_folder/'
        
    if not os.path.exists(flickr_folder):
        os.mkdir(flickr_folder) 
    
    flickr = FlickrAPI('9b0ccc40e4f15810510b5212a8e6d1c5', 'e07d2b15bf994f1e', cache=True)
    get_from_flickr(flickr, keyward, image_count, flickr_folder)
