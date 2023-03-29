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

#import helper libraries
import time
from urllib.parse import urlparse
import os
import requests
import io
from PIL import Image
import re
# using https://stackoverflow.com/questions/65199011/is-there-a-way-to-check-similarity-between-two-full-sentences-in-python
import spacy
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg") # large language model
import numpy as np
from scipy import interpolate

#custom patch libraries
import patch

class GoogleImageScraper():
    def __init__(self, webdriver_path, image_path, search_key="cat", number_of_images=1, token_name="cat", headless=True, min_resolution=(0, 0), max_resolution=(1920, 1080), max_missed=10):
        #check parameter types
        if (type(number_of_images)!=int):
            print("[Error] Number of images must be integer value.")
            return
        if not os.path.exists(image_path):
            print("[INFO] Image path not found. Creating a new folder.")
            os.makedirs(image_path)
            
        #check if chromedriver is installed
        if (not os.path.isfile(webdriver_path)):
            is_patched = patch.download_lastest_chromedriver()
            if (not is_patched):
                exit("[ERR] Please update the chromedriver.exe in the webdriver folder according to your chrome version:https://chromedriver.chromium.org/downloads")

        for i in range(1):
            try:
                #try going to www.google.com
                firefox_options = Options()
                if headless:
                    firefox_options.add_argument("--headless")
                driver = webdriver.Firefox(options=firefox_options)
                driver.set_window_size(1400,1050)
                driver.get("https://www.google.com")
            except Exception as e:
                #update chromedriver
                pattern = '(\d+\.\d+\.\d+\.\d+)'
                version = list(set(re.findall(pattern, str(e))))[0]
                is_patched = patch.download_lastest_chromedriver(version)
                if (not is_patched):
                    exit("[ERR] Please update the chromedriver.exe in the webdriver folder according to your chrome version:https://chromedriver.chromium.org/downloads")

        self.driver = driver
        self.search_key = search_key
        self.token_name = token_name
        self.number_of_images = number_of_images
        self.webdriver_path = webdriver_path
        self.image_path = image_path
        self.url = "https://www.google.com/search?q=%s&source=lnms&tbm=isch&sa=X&ved=2ahUKEwie44_AnqLpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947"%(search_key)
        self.headless=headless
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_missed = max_missed

    def find_image_urls(self):
        """
            This function search and return a list of image urls based on the search key.
            Example:
                google_image_scraper = GoogleImageScraper("webdriver_path","image_path","search_key",number_of_photos)
                image_urls = google_image_scraper.find_image_urls()

        """
        print("[INFO] Gathering image links")
        image_urls = [''] * self.number_of_images
        image_title_similarities = np.zeros(self.number_of_images)
        doc_s = nlp(self.search_key)
        missed_count = 0
        self.driver.get(self.url)
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "rg_i.Q4LuWd")))
        idx = 0
        elems = self.driver.find_elements(By.CLASS_NAME,"rg_i.Q4LuWd") # Find images in page
        Le = len(elems)
        while idx < self.number_of_images:
            nfails = 0
            while nfails < 3:
                try:
                    elem = elems[idx] # go to next image
                    elem.click() # click on image to open side bar
                    WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "l39u4d")))
                    side_bar = self.driver.find_element(By.CLASS_NAME,'l39u4d')
                    try:
                        image_urls[idx] = side_bar.find_element(By.CLASS_NAME,'n3VNCb.pT0Scc.KAlRDb').get_attribute('src')
                    except:
                        image_urls[idx] = side_bar.find_element(By.CLASS_NAME,'n3VNCb.pT0Scc').get_attribute('src')
                    main_img_title = side_bar.text.split('\n')[1]
                    if len(main_img_title) == 0:
                        main_img_title = side_bar.find_element(By.CLASS_NAME,'eYbsle.mKq8g.cS4Vcb-pGL6qe-fwJd0c').text
                    # Now we compare the title to the search prompt with a large language model to evaluate the similarity
                    doc_i = nlp(main_img_title)
                    image_title_similarities[idx] = doc_s.similarity(doc_i)
                    # If we get through without failing
                    break
                except: 
                    nfails += 1
            if idx%10 == 0 and idx >= 10: # run every 10
                simoothed = interpolate.splrep(np.arange(idx), image_title_similarities[0:idx],k=3,s=200)
            idx += 1
            # Check to see if scrolling is necessary
            if idx > Le:
                self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
                WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "rg_i.Q4LuWd")))
                elems = self.driver.find_elements(By.CLASS_NAME,"rg_i.Q4LuWd") # Find images in page
                Le = len(elems)

        self.driver.quit()
        print("[INFO] Google search ended")
        return image_urls

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
        print("[INFO] Saving image, please wait...")
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
                                filename = "%s_%s.%s"%(self.token_name,str(indx),image_from_web.format.lower())

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

