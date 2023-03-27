# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:02:06 2020

@author: OHyic

"""
#Import libraries
import os
import sys
import concurrent.futures
from GoogleImageScraper import GoogleImageScraper
from patch import webdriver_executable


def worker_thread(search_key, num_images,min_res,max_res,image_path,webdriver_path,keep_filenames=False,headless=True):
    image_scraper = GoogleImageScraper(
        webdriver_path, image_path, search_key, num_images, headless, min_res, max_res)
    image_urls = image_scraper.find_image_urls()
    image_scraper.save_images(image_urls, keep_filenames)

    # Release resources
    del image_scraper

if __name__ == "__main__":
    # Define file path
    webdriver_path = os.path.normpath(os.path.join(os.getcwd(), 'webdriver', webdriver_executable()))
    image_path = os.path.normpath(os.path.join(os.getcwd(), 'images/google-images'))

    # Add new search key into array ["cat","t-shirt","apple","orange","pear","fish"]
    if len(sys.argv) > 1:
        search_keys = sys.argv[1:]
    else:
        search_keys = list(set(["Inspector Lunge"]))

    # Parameters
    number_of_images = 500              # Desired number of images
    headless = True                     # True = No Chrome GUI
    min_resolution = (0, 0)             # Minimum desired image resolution
    max_resolution = (9999, 9999)       # Maximum desired image resolution
    max_missed = 1000                   # Max number of failed images before exit
    number_of_workers = 1               # Number of "workers" used
    keep_filenames = False              # Keep original URL image filenames

    #Run each search_key in a separate thread
    #Automatically waits for all threads to finish
    #Removes duplicate strings from search_keys
    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        executor.map(worker_thread, search_keys,number_of_images,min_resolution,max_resolution,image_path,webdriver_path,keep_filenames,headless)
