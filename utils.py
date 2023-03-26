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


def list_anime_characters(anime_name,image_path='',keep_filenames=False):
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
    for elem in elem_list:
        character_name = elem.text.partition("\n")[0]
        if ',' in character_name:
            lastfirst = character_name.partition(",")
            character_name = lastfirst[2] + lastfirst[0]
            if character_name[0] == " ": # extra space is generated
                character_name = character_name[1:]
        character_names.append(character_name)
        # Now we want to extract the image that comes with the character, as for minor characters we won't find one elsewhere.
        link_elem = elem.find_element(By.CLASS_NAME,'spaceit_pad').find_element(By.CSS_SELECTOR,'a')
        character_links.append(link_elem.get_attribute('href'))
    for chidx in range(len(character_names)):
        character_name = character_names[chidx]
        link = character_links[chidx]
        driver.get(link)
        wait = UI.WebDriverWait(driver, 3000)
        elem = driver.find_element(By.CLASS_NAME,'portrait-225x350.lazyloaded') # find main character image on the page
        image_url = elem.get_attribute('src')
        try:
            print("[INFO] Image url:%s"%(image_url))
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
                            filename = "%s%s.%s"%(character_name,'000',image_from_web.format.lower())

                        full_image_path = os.path.join(image_path, filename)
                        print(
                            f"[INFO] {character_name} \t {indx} \t Image saved at: {image_path}")
                        image_from_web.save(image_path)
                    except OSError:
                        rgb_im = image_from_web.convert('RGB')
                        rgb_im.save(image_path)
                    image_resolution = image_from_web.size
                    image_from_web.close()
        except Exception as e:
            print("[ERROR] Download failed: ",e)
            pass








if __name__ == '__main__':
    list_anime_characters('Monster')