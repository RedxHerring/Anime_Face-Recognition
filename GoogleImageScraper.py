# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:01:02 2020

@author: OHyic
"""
#import selenium drivers
from selenium import webdriver
# make sure geckodriver installed in default locaiton for OS. For linux installation the package manager should do its job here.
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager

#import helper libraries
from datetime import datetime
from urllib.parse import urlparse
import os
import requests
import io
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from itertools import compress


class GoogleImageScraper():
    def __init__(self, image_path, search_key="cat", number_of_images=1, token_name="cat", driver=None, headless=True, min_resolution=(0, 0), max_resolution=(1920, 1080), max_missed=10):
        #check parameter types
        if (type(number_of_images)!=int):
            print("[Error] Number of images must be integer value.")
            return
        if not os.path.exists(image_path):
            print("[INFO] Image path not found. Creating a new folder.")
            os.makedirs(image_path)

        if driver is None:
            firefox_options = Options()
            if headless:
                firefox_options.add_argument("--headless")
            driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()),options=firefox_options)
            driver.set_window_size(1400,1050)

        self.driver = driver
        self.search_key = search_key
        self.token_name = token_name
        self.number_of_images = number_of_images
        self.image_path = image_path
        self.url = "https://www.google.com/search?q=%s&source=lnms&tbm=isch&sa=X&ved=2ahUKEwie44_AnqLpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947"%(search_key)
        self.headless=headless
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_missed = max_missed

    # Load or reload full page
    def loadnscroll(self,class_name):
        print(f"[INFO] Loading {self.url}")
        max_wait_time = 4 # time to wait for images to load, in seconds
        self.driver.get(self.url)
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, class_name)))
        elems = self.driver.find_elements(By.CLASS_NAME,class_name) # Find images in page
        Le = len(elems)
        loaded_more = False
        while Le < self.number_of_images*1.2: # add buffer since images will be filtered out later
            self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, class_name)))
            elems = self.driver.find_elements(By.CLASS_NAME,class_name) # Find images in page
            t1 = datetime.now()
            while Le == len(elems): # check if new elems have loaded
                t2 = datetime.now()
                elems = self.driver.find_elements(By.CLASS_NAME,class_name) # Find images in page
                if (t2-t1).seconds >= max_wait_time: # no more images will load
                    if loaded_more: 
                        break
                    else: # Haven't clicked button at the bottom yet
                        elemss = self.driver.find_elements(By.CLASS_NAME,"mye4qd") # button at bottom
                        idx = 0
                        while idx < len(elemss): # one is "Show more results" button
                            try:
                                elemss[idx].click()
                                idx += 1
                            except:
                                idx += 1
                        loaded_more = True
                        t1 = datetime.now() # reset clock
            if Le == len(elems): # no more images will load even after waiting
                break
            Le = len(elems)
        # Find images in now fully-loaded page
        elems = self.driver.find_elements(By.CLASS_NAME,class_name) 
        self.driver.execute_script("window.scrollTo(0,0)") # scroll back to top
        return elems

    # Find links to google images in search results 
    def find_image_urls(self):
        """
            This function search and return a list of image urls based on the search key.
            Example:
                google_image_scraper = GoogleImageScraper("webdriver_path","image_path","search_key",number_of_photos)
                image_urls = google_image_scraper.find_image_urls()

        """

        # Using https://towardsdatascience.com/semantic-textual-similarity-83b3ca4a840e
        # Load the pre-trained model
        model = SentenceTransformer('stsb-mpnet-base-v2')

        # Also make a list of strings to check for in image titles to remove fanart, etc
        reject_strings = ["danbooru","cosplay","poster","wallpaper","stl"]

        print("[INFO] Gathering image links")

        sentence1_emb = model.encode(self.search_key.lower(), show_progress_bar=False)
        sentence1_emb = sentence1_emb/np.sqrt(np.sum(sentence1_emb**2))
        elems = self.loadnscroll("rg_i.Q4LuWd") # load as much of the page as possible
        Le = len(elems)
        dfimg = pd.DataFrame({'url': [''] * Le,
                              'title': [''] * Le,
                              'similarity': [0] * Le})
        sim_thresh = .5
        idx = 0
        winlen = min(25,Le)
        nskips = 0
        have_reloaded = False
        while idx < Le:
            nfails = 0
            while nfails < 3:
                try:
                    elem = elems[idx] # go to next image
                    elem.click() # click on image to open side bar
                    WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "l39u4d")))
                    side_bar = self.driver.find_element(By.CLASS_NAME,'l39u4d')
                    try:
                        dfimg.at[idx,'url'] = side_bar.find_element(By.CLASS_NAME,'r48jcc.pT0Scc.iPVvYb').get_attribute('src')
                    except:
                        dfimg.at[idx,'url'] = side_bar.find_element(By.CLASS_NAME,'pT0Scc').get_attribute('src')
                    print(f"[INFO] {self.search_key} \t #{idx} \t {dfimg['url'][idx]}")
                    dfimg.at[idx,'title'] = side_bar.text.split('\n')[1]
                    if len(dfimg['title'][idx]) == 0:
                        dfimg.at[idx,'title'] = side_bar.find_element(By.CLASS_NAME,'eYbsle.mKq8g.cS4Vcb-pGL6qe-fwJd0c').text.lower()
                    # Now we compare the title to the search prompt with a large language model to evaluate the similarity
                    sentence2_emb = model.encode(dfimg['title'][idx], show_progress_bar=False)
                    sentence2_emb = sentence2_emb/np.sqrt(np.sum(sentence2_emb**2))
                    if any(rjstr in dfimg['title'][idx] for rjstr in reject_strings):
                        dfimg.at[idx,'similarity'] = 0 # not looking for content with this string in its title
                        print("[Info] Image rejected due to rejected content in image title.")
                    else:   
                        dfimg.at[idx,'similarity'] = sentence1_emb.T@sentence2_emb # take inner product
                        print(f"[INFO] '{self.search_key}' has inner product of {dfimg['similarity'][idx]} with '{dfimg['title'][idx]}'")
                    # If we get through without failing
                    break
                except: 
                    nfails += 1
            idx += 1
            if nfails > 3:
                nskips += 1
            if nskips > 3 and not have_reloaded: # something wrong, reload page
                print ('[NOTICE] Reloading due to consecutive fails')
                elems = self.loadnscroll("rg_i.Q4LuWd") # load as much of the page as possible
                idx = idx - nskips
                nskips = 0 # reset
                have_reloaded = True # don't wanna get caught in an infinite loop
        simoothed = np.convolve(dfimg['similarity'][0:idx], np.ones(winlen), 'same') / winlen
        self.driver.quit()
        dfimg.to_csv(os.path.join(self.image_path,self.token_name+'-results.csv'))
        # Keep all links that are above the threshold or are surrounded by links above the threshold, except for 0s removed due to content words
        keep_rows = np.logical_and(np.logical_or(dfimg["similarity"]>sim_thresh, simoothed>sim_thresh), dfimg["similarity"]>0)
        image_urls = list(compress(dfimg['url'].to_list(),keep_rows))
        unique_urls = list(set(image_urls)) # sets remove any duplicates
        print("[INFO] Google search ended")
        return unique_urls[0:min(len(unique_urls)-1,self.number_of_images)]

    def save_images(self,image_urls, keep_filenames):
        some_failed = False
        #save images into file directory
        """
            This function takes in an array of image urls and save it into the given image path/directory.
            Example:
                google_image_scraper = GoogleImageScraper("webdriver_path","image_path","search_key",number_of_photos)
                image_urls=["https://example_1.jpg","https://example_2.jpg"]
                google_image_scraper.save_images(image_urls)

        """
        num_digits = int(np.ceil(np.log10(self.number_of_images)))
        print("[INFO] Saving images, please wait...")
        for indx,image_url in enumerate(image_urls):
            try:
                print("[INFO] Image url:%s"%(image_url))
                search_string = ''.join(e for e in self.search_key if e.isalnum())
                image = requests.get(image_url,timeout=5)
                if image.status_code == 200:
                    with Image.open(io.BytesIO(image.content)) as image_from_web:
                        try:
                            if (keep_filenames):
                                #extact filename without extension from URL
                                o = urlparse(image_url)
                                image_url = o.scheme + "://" + o.netloc + o.path
                                name = os.path.splitext(os.path.basename(image_url))[0]
                                #join filename and extension
                                filename = "%s.%s"%(name,image_from_web.format.lower())
                            else:
                                indxstr = str(indx)
                                indxstr = '0'*(num_digits-len(indxstr)) + indxstr
                                filename = "%s_%s.%s"%(self.token_name,indxstr,image_from_web.format.lower())

                            image_path = os.path.join(self.image_path, filename)
                            print(
                                f"[INFO] {self.search_key} \t {indx} \t Image saved at: {image_path}")
                            image_from_web.save(image_path)
                        except OSError:
                            rgb_im = image_from_web.convert('RGB')
                            rgb_im.save(image_path)
                        image_resolution = image_from_web.size
                        if image_resolution != None:
                            if image_resolution[0]<self.min_resolution[0] or image_resolution[1]<self.min_resolution[1] or image_resolution[0]>self.max_resolution[0] or image_resolution[1]>self.max_resolution[1]:
                                image_from_web.close()
                                os.remove(image_path)

                        image_from_web.close()
            except Exception as e:
                print("[ERROR] Download failed: ",e)
                some_failed = True
                pass
        print("--------------------------------------------------")
        if some_failed:
            print("[INFO] Downloads completed. Please note that some photos were not downloaded as they were not in the correct format (e.g. jpg, jpeg, png)")
        else:
            print("[INFO] All downloads completed successfully.") 

