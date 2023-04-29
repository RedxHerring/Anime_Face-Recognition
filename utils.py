# Here there will be a number of short scripts
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager

import requests
import cv2
import os
import pandas as pd
import wget
import glob
from runGoogleImagScraper import parallel_worker_threads

def char_is_num(x):
    if(x >= '0' and x <= '9'):
        return True
    else:
        return False

# This function takes as input an anime name and wills earch for it on myanimelist,
# and then find all the characters listed for that anime and save names and nicknames
# in a csv, as well as saving an image of the character in a given directory
def list_anime_characters(anime_name,images_path='',keep_filenames=False):
    os.makedirs(images_path,exist_ok=True)
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()),options=firefox_options)
    # Using https://stackoverflow.com/questions/63232160/cannot-locate-search-bar-with-selenium-in-python
    driver.get('https://myanimelist.net/anime.php')
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "q")))
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
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'js-anime-character-table')))
    elem_list = driver.find_elements(By.CLASS_NAME,'js-anime-character-table')
    character_links = []
    character_names = []
    character_types = []
    # Loop though list to get character's primary name and link to character's page
    print('[INFO] Collecting characters')
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
        character_types.append(elem.find_elements(By.CLASS_NAME,'spaceit_pad')[1].text)
        character_links.append(link_elem.get_attribute('href'))

    # Create table with character name and links, as well as any alternative names.
    d = {'Name': character_names, 'Character_Type': character_types, 'Other_Names': ['']*len(character_names), 'Link': character_links, 'Image_Link': character_links}
    # Note we reused variables to initialize the table values,a nd we will modify them as we loop through
    df = pd.DataFrame(data=d)
    for chidx in range(len(character_names)): # Loop through all characters in the anime
        character_name = character_names[chidx]
        link = character_links[chidx]
        image_path = os.path.join(images_path,character_name.replace(" ","_"))
        os.makedirs(image_path,exist_ok=True)
        driver.get(link)
        cidx = 0
        while cidx < 3:
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'title-name.h1_bold_none')))
                # Now that we're on the page, we try to get alternative character names, which will be in the title in ""
                elem = driver.find_element(By.CLASS_NAME,'title-name.h1_bold_none')
                names_list = elem.text.split('"')
                if len(names_list) > 1:
                    df.Other_Names[chidx] = names_list[1]
                break
            except:
                cidx += 1
        # Now we want to extract the image that comes with the character, as for minor characters we won't find one elsewhere.
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'portrait-225x350.lazyloaded')))
        elem = driver.find_element(By.CLASS_NAME,'portrait-225x350.lazyloaded') # find main character image on the page
        image_url = elem.get_attribute('src')
        df.Image_Link[chidx] = image_url
        # Download with wget
        print(f'\n[INFO] Downloading image for {df.Name[chidx]}')
        wget.download(image_url,out=image_path)
    # With the dataframe table complete, save it to a csv
    df.to_csv(anime_name.replace(' ','_') + '-Characters.csv')

# This function takes as input the anme of a csv as generated by list_anime_characters,
# and then runs other functions to download google iamges for each character.
def get_character_images(anime_file,images_path=''):
    df = pd.read_csv(anime_file)
    anime_name = anime_file.split('-')[0]
    append_str = f" {anime_name} anime"
    for idx in df.index:
        image_path = os.path.join(images_path,df.Name[idx].replace(" ","_"))
        character_names = [df.Name[idx]]
        if type(df.Other_Names[idx]) == str:
            additional_keys = df.Other_Names[idx].split(',')
            character_names.extend(additional_keys)
        search_keys = [s + append_str for s in character_names] # append anime name and "anime" in search text
        token_names = [s.replace(" ","_") for s in character_names]
        parallel_worker_threads(search_keys,token_names=token_names,imgs_path=image_path,num_images=500,maxmissed=1000)

def is_gray(imgpath):
    threshold = 7
    img = cv2.imread(imgpath)
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv_img[:, :, 1]
    mean_sat = sat.mean()
    if mean_sat < threshold:
        return True
    else:
        return False

def check_gray(dir):
    for img in os.listdir(dir):
        try:
            print(img + ': ' + str(is_gray(dir + img)))
        except:
            pass

def remove_grayscale_images(anime_file,images_path=''):
    if images_path[0] != '/':
        images_path = os.path.normpath(os.path.join(os.getcwd(),images_path))
    df = pd.read_csv(anime_file)
    for idx in df.index:
        image_path = os.path.join(images_path,df.Name[idx].replace(" ","_"))
        for file in os.listdir(image_path):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".webp"):
                full_name = os.path.join(image_path,file)
                if is_gray(full_name):
                    os.remove(full_name)

def overlapped_area(a, b):  # returns intersecting area >= 0
    # From https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    # a and b express rectangles as a = [a_xmin, a_ymin, a_xmax, a_ymax]
    dx =  max(min(a[2],b[2]) - max(a[0],b[0]), 0)
    dy =  max(min(a[3],b[3]) - max(a[1],b[1]), 0)
    return dx*dy

def crop_faces(img_name, img=None, detector=None, cropped_dir='Images/cropped-images',idx0=0):
    if detector is None:
        detector = cv2.FaceDetectorYN.create(
            "models/fd_yunet.onnx",
            "",
            input_size=(320, 320),
            score_threshold=.3,
            nms_threshold=.4
        )
    if img is None: # need to load image
        if not os.path.exists(img_name):
            print('[INFO] Image file does not exist, skipping')
            return detector, idx0
        fullname, file_extension = os.path.splitext(img_name)
        filename = os.path.basename(fullname)
        character_name = os.path.basename(os.path.dirname(filename))
        images_path = os.path.join(cropped_dir,character_name)
        os.makedirs(images_path,exist_ok=True)
        img = cv2.imread(img_name)
    else:
        images_path = os.path.join(cropped_dir,img_name)
    m,n,_ = img.shape
    detector.setInputSize((n,m))
    try:
        faces = detector.detect(img)
    except: # The image is simply too hot to handle, this is a rare bug
        print("[INFO] Image detect failed, skipping")
        detector = None # just in case something got wonky since it's uncharted territory
        return detector, idx0
    idx = 0 # initialize in case the for loop is skipped
    if faces[1] is not None:
        # First we need to check for and remove cases of boxes within boxes.
        # Setting a lower nms_threshold would remove the outer box, but if that larger box is around two smaller boxes,
        # it likely means the larger box is detecting features from two seperate but nearby heads
        max_overlap = .4 # max fraction of face box that can intersect another box before we have a problem
        coords = faces[1][:,0:4].astype(np.int32)
        # Some value of coords may be negative numbers if it interpolates the face as extending outside of the image.
        # We will leave this for now, as we want a good idea of face sizes for the sorting process.
        Nf = len(coords)
        if Nf > 1:
            rects = np.vstack((coords[:,0],coords[:,1],coords[:,0]+coords[:,2],coords[:,1]+coords[:,3])).T
            areas = np.prod(coords[:,2:],axis=1)
            # Initialize variable to store ares and boolean mask array
            overlapped_areas = np.zeros((Nf,Nf))
            keep_rect = np.full((Nf), True)
            for idx in range(Nf):
                for jdx in range(idx,Nf):
                    if idx == jdx:
                        continue
                    overlapped_areas[idx,jdx] = overlapped_area(rects[idx,:],rects[jdx,:])
                    overlapped_areas[jdx,idx] = overlapped_areas[idx,jdx]
                ovlp_wrt_j = overlapped_areas[idx,:]/areas
                if sum(ovlp_wrt_j>max_overlap) > 1: # this box contains mutliple faces
                    keep_rect[idx] = False
                elif sum(ovlp_wrt_j>max_overlap) == 1:
                    keep_rect[ovlp_wrt_j>max_overlap] = False
            coords = coords[keep_rect,:]
        # With the sorting process done, we need to remove any negative values.
        coords = np.maximum(0,coords)
        # There may still be values that are too large, but it's more convenient to sort this out within the loop
        Nf = len(coords)
        for idx in range(Nf):
            x1, y1, w, h = coords[idx,0:4]
            if w == 0 or h == 0:
                continue
            # First save cropped version as determined by Yunet.
            # This way we can compare the images with other cropped ones
            x2 = x1 + w
            if x2 > n:
                x2 = n
            y2 = y1 + h
            if y2 > m:
                y2 = m
            imgi = img[y1:y2, x1:x2, :]
            namei =  os.path.join(images_path,filename+'-'+str(idx0+idx)+'.png')
            cv2.imwrite(namei,imgi)
            # Now get a square crop to use with CNN, which we can easily scale as needed.
            # Start by enlarging by 10% in all directions, to get full chin and ears, and more hair.
            ws = int(1.2*max(w,h))
            # Expand borders in x direction
            dw = int((ws-w)/2)
            x1 = x1 - dw
            if x1 < 0:
                x1 = 0
            x2 = x1 + ws
            if x2 > n:
                x2 = n
                ws = x2 - x1 + 1 # shrink border
            # Expand borders in y direction
            hs = ws
            dh = int((hs-h)/2)
            y1 = y1 - dh
            if y1 < 0:
                y1 = 0
            y2 = y1 + hs
            if y2 > m:
                y2 = m
                hs = y2 - y1 + 1
            # Now shrink x again if necessary
            if hs < ws:
                dw = int((ws-hs)/2)
                ws = hs
                x1 = x1 + dw
                x2 = x1 + ws
            imgs = img[y1:y1+hs, x1:x1+ws, :]
            names =  os.path.join(images_path,filename+'-square'+str(idx0+idx)+'.png')
            cv2.imwrite(names,imgs)
    return detector, idx0+idx # so we don't have to re-initialize next time

def download_models():
    link = 'https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx'
    r = requests.get(link)
    download_path = 'models'
    os.makedirs(download_path,exist_ok=True)
    file_name = 'fd_yunet.onnx'
    with open(os.path.join(download_path, file_name), 'wb') as fd:
        fd.write(r.content)

def crop_faces_in_video(video_path):
    cap = cv2.VideoCapture('linusi.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()

def crop_faces_all():
    dirs = glob.glob('Images/google-images-original/*')
    for dir in dirs:
        crop_dir = os.path.join('Images/google-images-cropped/',dir[30:])
        print(crop_dir)
        imgs = glob.glob(dir + '/*.jpeg')
        for img in imgs:
            print('[INFO] Cropping ' + img)
            crop_faces(img, cropped_dir=crop_dir)


if __name__ == '__main__':
    list_anime_characters('Monster','Images/myanimelist-images-original')
    # get_character_images("Monster-Characters.csv",'Images/google-images')
    # load_image('Images/google-images/Adolf_Junkers/Adolf_Junkers_0.webp')
    # remove_grayscale_images("Monster-Characters.csv",'Images/google-images')
    # check_gray('Images/google-images/Anna_Liebert/')
    # download_models()
    # image_name = 'Images/google-images-original/Robbie/Robbie_25.jpeg'
    # detector = crop_faces(image_name,cropped_dir='Images/google-images-cropped/Robbie')
    # crop_faces_all()
