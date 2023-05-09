# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:02:06 2020

@author: OHyic

"""
#Import libraries
import concurrent.futures
from GoogleImageScraper import GoogleImageScraper
import argparse


def worker_thread(search_key,token_name):
    image_scraper = GoogleImageScraper(image_path=images_path, search_key=search_key, number_of_images=number_of_images, token_name=token_name, 
                                       headless=headless, min_resolution=min_resolution, max_resolution=max_resolution, max_missed=max_missed,
                                       reject_strings=reject_strings, sim_thresh=sim_thresh)
    image_urls = image_scraper.find_image_urls()
    image_scraper.save_images(image_urls, keep_filenames)
    #Release resources
    del image_scraper

def parallel_worker_threads(search_keys=['cat'],token_names=['image'],imgs_path='google-images',num_images=20,maxmissed=10,keepfilenames=False,
                            min_res=(0,0),max_res=(9999,9999),isheadless=True,reject_strs=None,simthresh=.5):
    # First make sure variable types are set correctly
    if type(search_keys) is str:
        search_keys = [search_keys]
    if type(token_names) is str:
        token_names = [token_names]
    
    # We unfortunately only want to iterate through search_keys, with the rest being global variables used by worker_thread
    global images_path
    images_path = imgs_path
    global number_of_images
    number_of_images = num_images
    global max_missed
    max_missed = maxmissed
    global keep_filenames
    keep_filenames = keepfilenames
    global min_resolution
    min_resolution = min_res
    global max_resolution
    max_resolution = max_res
    global headless
    headless = isheadless
    global reject_strings
    reject_strings = reject_strs
    global sim_thresh
    sim_thresh = simthresh

    number_of_workers = len(search_keys) # Number of "workers" used
    if len(token_names) != number_of_workers:
        token_names = [token_names[0]]*len(search_keys)
    #Run each search_key in a separate thread
    #Automatically waits for all threads to finish
    #Removes duplicate strings from search_keys
    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        executor.map(worker_thread, search_keys, token_names)



if __name__ == "__main__":
    # Using fork of original here: https://github.com/rundfunk47/Google-Image-Scraper/blob/master/main.py
    # Define the command line arguments that the program should accept
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--search-keys', action = 'append', help='the search keys to use for scraping images', required=True)
    parser.add_argument('-n', '--number-of-images', type=int, help='the number of images to scrape', default=20)
    parser.add_argument('-H', '--headless', action='store_true', help='when on, the app will not run in headless mode', default=True)
    parser.add_argument('-p', '--images-path', help='the desired image directory', default='google-images')
    parser.add_argument('-m', '--max-missed', type=int, help='the maximum number of failed images before exiting', default=10)
    parser.add_argument('-k', '--keep-filenames', action='store_true', help='keep the original filenames of the images', default=False)
    parser.add_argument('-t', '--token_names', help='the filename to use when storing the files. I.e. tokenname "jwa" will store files "jwa_1.jpg", "jwa_2.jpg" and so on. this has no effect if --keep-filenames is True', default=None)
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Use the values from the command line arguments for the parameters
    search_keys = args.search_keys
    number_of_images = args.number_of_images
    headless = args.headless
    images_path = args.images_path
    max_missed = args.max_missed
    keep_filenames = args.keep_filenames

    # If the token_name argument is not provided, set it to the same value as the search_key argument
    if args.token_name is None:
        token_names = args.search_keys
    else:
        token_names = args.token_names

    

    # Parameters
    min_resolution = (0, 0)              # Minimum desired image resolution
    max_resolution = (9999, 9999)        # Maximum desired image resolution
    number_of_workers = len(search_keys) # Number of "workers" used

    #Run each search_key in a separate thread
    #Automatically waits for all threads to finish
    #Removes duplicate strings from search_keys
    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        executor.map(worker_thread, search_keys)
