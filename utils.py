# Here there will be a number of short scripts 
from PIL import Image
import numpy as np
import time
from selenium import webdriver
# make sure geckodriver installed in default locaiton for OS. For linux installation the package manager should do its job here.
from selenium.webdriver.firefox.options import Options 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import selenium.webdriver.support.ui as UI
import requests
import cv2
import os
import io
import pandas as pd
import wget

# Function to return boolean True if image is black&white, and False otherwise
def is_black_n_white(img):


    kernel = np.ones((2,2),np.uint8)

    # load image

    img = cv2.imread("2.jpg")

    # Convert BGR to HSV

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of black color in HSV

    lower_val = np.array([0,0,0])

    upper_val = np.array([179,100,130])

    # Threshold the HSV image to get only black colors

    mask = cv2.inRange(hsv, lower_val, upper_val)

    # Bitwise-AND mask and original image

    res = cv2.bitwise_and(img,img, mask= mask)

    # invert the mask to get black letters on white background

    res2 = cv2.bitwise_not(mask)

    # display image

    cv2.imshow("img", res)

    cv2.imshow("img2", res2)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

def char_is_num(x):
    if(x >= '0' and x <= '9'):
        return True
    else:
        return False

def list_anime_characters(anime_name,images_path='',keep_filenames=False):
    os.makedirs(images_path,exist_ok=True)
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    driver = webdriver.Firefox(options=firefox_options)
    # Using https://stackoverflow.com/questions/63232160/cannot-locate-search-bar-with-selenium-in-python
    driver.get('https://myanimelist.net/anime.php')
    wait = UI.WebDriverWait(driver, 3000)
    elem = driver.find_element(By.ID,"q")
    elem.send_keys(anime_name)
    elem.send_keys(Keys.ENTER)
    # Now we need to go through the search results and find the right one. 
    time.sleep(5)
    # Class contains spaces which we replace with dots
    elem_list = driver.find_elements(By.CLASS_NAME,'hoverinfo_trigger.fw-b.fl-l')
    for elem in elem_list: # Loop through search results
        if elem.text == anime_name:
            # get the attribute value
            link = elem.get_attribute('href')
            break
    driver.get(link + '/characters')
    # Now we're on the characters page for the anime we want.
    wait = UI.WebDriverWait(driver, 3000)
    elem_list = driver.find_elements(By.CLASS_NAME,'js-anime-character-table')
    character_links = []
    character_names = []
    # Loop though list to get character's primary name and link to character's page
    for elem in elem_list:
        character_name = elem.text.partition("\n")[0]
        if ',' in character_name:
            lastfirst = character_name.partition(",")
            character_name = lastfirst[2] + "_" + lastfirst[0]
            if character_name[0] == " ": # extra space is generated
                character_name = character_name[1:]
        character_name = character_name.replace(" ","_")
        while character_name in character_names: # Character name already present
            if char_is_num(character_name[-1]):
                character_name = character_name[0:-1] + str(int(character_name[-1])+1)
            else:
                character_name = character_name + '_1'
        character_names.append(character_name)
        # Use class and css to find character page link.
        link_elem = elem.find_element(By.CLASS_NAME,'spaceit_pad').find_element(By.CSS_SELECTOR,'a')
        character_links.append(link_elem.get_attribute('href'))
    
    # Create table with character name and links, as well as any alternative names.

    for chidx in range(len(character_names)): # Loop through all characters in the anime
        character_name = character_names[chidx]
        link = character_links[chidx]
        image_path = os.path.join(images_path,character_name)
        os.makedirs(image_path,exist_ok=True)
        driver.get(link)
        time.sleep(5)
        # Now that we're on the page, we try to get alternative character names, which will be in the title in ""
        elem = driver.find_element(By.CLASS_NAME,'title-name.h1_bold_none')
        elem.text.partition('"')
        # Now we want to extract the image that comes with the character, as for minor characters we won't find one elsewhere.
        elem = driver.find_element(By.CLASS_NAME,'portrait-225x350.lazyloaded') # find main character image on the page
        image_url = elem.get_attribute('src')
        # Download with wget
        wget.download(image_url,out=image_path)


if __name__ == '__main__':
    list_anime_characters('Monster','Images/original-images')