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
from runGoogleImagScraper import worker_thread
from patch import webdriver_executable
import concurrent.futures


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
            character_name = lastfirst[2] + " " + lastfirst[0]
            if character_name[0] == " ": # extra space is generated
                character_name = character_name[1:]
        character_name = character_name
        while character_name in character_names: # Character name already present
            if char_is_num(character_name[-1]):
                character_name = character_name[0:-1] + str(int(character_name[-1])+1)
            else:
                character_name = character_name + " 1"
        character_names.append(character_name)
        # Use class and css to find character page link.
        link_elem = elem.find_element(By.CLASS_NAME,'spaceit_pad').find_element(By.CSS_SELECTOR,'a')
        character_links.append(link_elem.get_attribute('href'))
    
    # Create table with character name and links, as well as any alternative names.
    d = {'Name': character_names, 'Other_Names': ['']*len(character_names), 'Link': character_links, 'Image_Link': character_links}
    # Note we reused variables to initialize the table values,a nd we will modify them as we loop through
    df = pd.DataFrame(data=d)
    for chidx in range(len(character_names)): # Loop through all characters in the anime
        character_name = character_names[chidx]
        link = character_links[chidx]
        image_path = os.path.join(images_path,character_name.replace(" ","_"))
        os.makedirs(image_path,exist_ok=True)
        driver.get(link)
        time.sleep(5)
        # Now that we're on the page, we try to get alternative character names, which will be in the title in ""
        elem = driver.find_element(By.CLASS_NAME,'title-name.h1_bold_none')
        names_list = elem.text.split('"')
        if len(names_list) > 1:
            df.Other_Names[chidx] = names_list[1]
        # Now we want to extract the image that comes with the character, as for minor characters we won't find one elsewhere.
        elem = driver.find_element(By.CLASS_NAME,'portrait-225x350.lazyloaded') # find main character image on the page
        image_url = elem.get_attribute('src')
        df.Image_Link[chidx] = image_url
        # Download with wget
        wget.download(image_url,out=image_path)
    # With the dataframe table complete, save it to a csv
    df.to_csv(anime_name.replace(' ','_') + '-Characters.csv')


def get_character_images(anime_file,images_path=''):
    df = pd.read_csv(anime_file)
    # Define file path
    webdriver_path = os.path.normpath(os.path.join(os.getcwd(), 'webdriver', webdriver_executable()))
    # Parameters
    number_of_images = 500              # Desired number of images
    headless = True                     # True = No Chrome GUI
    min_resolution = (0, 0)             # Minimum desired image resolution
    max_resolution = (9999, 9999)       # Maximum desired image resolution
    max_missed = 1000                   # Max number of failed images before exit
    
    keep_filenames = False              # Keep original URL image filenames
    for idx in df.index:
        image_path = os.path.join(images_path,df.Name[idx].replace(" ","_"))
        search_keys = [df.Name[idx]]
        if type(df.Other_Names[idx]) == str:
            additional_keys = df.Other_Names[idx].split(',')
            search_keys.extend(additional_keys)
        for search_key in search_keys:
            worker_thread(search_key,number_of_images,min_resolution,max_resolution,image_path,webdriver_path,keep_filenames,headless)



if __name__ == '__main__':
    # list_anime_characters('Monster','Images/original-images')
    get_character_images("Monster-Characters.csv",'Images/google-images')

