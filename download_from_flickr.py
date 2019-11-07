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
    
    image_count = 1000; keyward = 'car'
    flickr_folder = '/home/valdis/Desktop/tfdataset/flickr_folder/'
        
    if not os.path.exists(flickr_folder):
        os.mkdir(flickr_folder) 
    
    flickr = FlickrAPI('private_key', 'add_to_key', cache=True)
    get_from_flickr(flickr, keyward, image_count, flickr_folder)
