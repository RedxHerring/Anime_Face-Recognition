# Here there will be a number of short scripts
import numpy as np
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.html5.application_cache import ApplicationCache
from webdriver_manager.firefox import GeckoDriverManager

import requests
from PIL import Image
from urllib.parse import urlparse
import io
import cv2
import os
import shutil
import pandas as pd
import glob
from configparser import ConfigParser
from itertools import compress
from runGoogleImagScraper import parallel_worker_threads
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

def char_is_num(x):
    if(x >= '0' and x <= '9'):
        return True
    else:
        return False


def download_image(image_url, image_path="", filename=None):
    if filename is None:
        keep_filenames = True
    else:
        keep_filenames = False
    image = requests.get(image_url,timeout=5)
    if image.status_code == 200:
        with Image.open(io.BytesIO(image.content)) as image_from_web:
            try:
                if keep_filenames:
                    #extact filename without extension from URL
                    o = urlparse(image_url)
                    image_url = o.scheme + "://" + o.netloc + o.path
                    name = os.path.splitext(os.path.basename(image_url))[0]
                    #join filename and extension
                    filename = "%s.%s"%(name,image_from_web.format.lower())

                image_path = os.path.join(image_path, filename)
                print(
                    f"[INFO] Image saved at: {image_path}")
                image_from_web.save(image_path)
            except OSError:
                rgb_im = image_from_web.convert('RGB')
                rgb_im.save(image_path)
            image_from_web.close()

def imma_human(driver,class_name):
    try:
        WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.CLASS_NAME, class_name)))
        return driver
    except: # We likely have been accused of being a bot
        driver.find_element(By.CLASS_NAME,'g-recaptcha').click()
        print("[INFO] Recursing to pass Turing test")
        return imma_human(driver,class_name)


# This function takes as input an anime name and will search for it on myanimelist,
# and then find all the characters listed for that anime and save names and nicknames
# in a csv, as well as saving an image of the character in a given directory
def list_anime_characters(anime_name,images_path='',keep_filenames=False):
    os.makedirs(images_path,exist_ok=True)
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()),options=firefox_options)
    app_cache = ApplicationCache(driver)
    app_cache.DOWNLOADING = 3
    # Using https://stackoverflow.com/questions/63232160/cannot-locate-search-bar-with-selenium-in-python
    driver.get('https://myanimelist.net/anime.php')
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "q")))
    elem = driver.find_element(By.ID,"q")
    elem.send_keys(anime_name)
    elem.send_keys(Keys.ENTER)
    # Now we need to go through the search results and find the right one.
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'hoverinfo_trigger.fw-b.fl-l')))
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
        # First conver "Last, First" into "First Last" since commas are separators in .csv
        if ',' in character_name:
            lastfirst = character_name.partition(",")
            character_name = lastfirst[2] + " " + lastfirst[0]
            if character_name[0] == " ": # extra space is generated
                character_name = character_name[1:]
        # Check if character name already present, can happen for small side character names
        while character_name in character_names:
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
        driver = imma_human(driver,class_name='title-name.h1_bold_none')
        # Now that we're on the page, we try to get alternative character names, which will be in the title in ""
        elem = driver.find_element(By.CLASS_NAME,'title-name.h1_bold_none')
        names_list = elem.text.split('"')
        if len(names_list) > 1:
            df.Other_Names[chidx] = names_list[1].replace('"','') # for multiple names the "" get dragged along for some reason
        # Now we want to extract the image that comes with the character, as for minor characters we won't find one elsewhere.
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'portrait-225x350.lazyloaded')))
        elem = driver.find_element(By.CLASS_NAME,'portrait-225x350.lazyloaded') # find main character image on the page
        image_url = elem.get_attribute('src')
        df.Image_Link[chidx] = image_url
        # Download with wget
        print(f'\n[INFO] Downloading image for {df.Name[chidx]}')
        if keep_filenames:
            download_image(image_url, image_path)
        else:
            download_image(image_url, image_path,character_name.replace(" ","_")+".png")
    # With the dataframe table complete, save it to a csv
    df.to_csv(anime_name.replace(' ','_') + '-Characters.csv')
    driver.quit()


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


def files_in_dir(files_dir,ftypes=['png','jpg','jpeg','webp']):
    files_list = []
    for ftype in ftypes:
        files_list.extend(glob.glob(os.path.join(files_dir,'*.'+ftype)))
    return sorted(files_list)


def remove_grayscale_images(anime_file,images_path=''):
    if images_path[0] != '/':
        images_path = os.path.normpath(os.path.join(os.getcwd(),images_path))
    df = pd.read_csv(anime_file)
    for idx in df.index:
        image_path = os.path.join(images_path,df.Name[idx].replace(" ","_"))
        image_files = files_in_dir(image_path)
        for file in image_files:
            full_name = os.path.join(image_path,file)
            if is_gray(full_name):
                os.remove(full_name)


def overlapped_area(a, b):  # returns intersecting area >= 0
    # From https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    # a and b express rectangles as a = [a_xmin, a_ymin, a_xmax, a_ymax]
    dx =  max(min(a[2],b[2]) - max(a[0],b[0]), 0)
    dy =  max(min(a[3],b[3]) - max(a[1],b[1]), 0)
    return dx*dy


def is_single_color(img):
    try:
        if np.std(img)<5 or np.std(img)/(np.max(img) - np.min(img))<.1:
            return True
    except: # something weird happened
        return True
    return False


def crop_faces(img_name="input_img.png", img=None, detector=None, cropped_dir='Images/cropped-images', idx0=0, score_threshold=.3, min_dim=16, save_rect=True, 
               save_square=True, return_faces=False, do_filtering=True, imres=(None,None), best_only=False):
    '''
    INPUTS
    img_name - full path to input image if img is None, or output name if img is input image
    img - input image array
    detector - instance of cv2.FaceDetectorYN if we want to customize it
    cropped_dir - output directory for images, or parent directory of subdirectory for images if image_name comes from a directory for a character name
    idx0 - index of image within that image, can be used if multiple images with the same img_name will be used so we can iterate indx0 across function calls
    score_threshold - score (~probabaility) for a face deteciton to pass as valid
    min_dim - minimum dimension of a detected face, for it to be considered valid
    save_rect - boolean to determine whether or not to save the crop from cv2
    save_square - boolean to determine whether to save larger modified square crop
    return_faces - return extra variables to use, including faces tuple and list of cropped images
    do_filtering - check if face image is effectively single-color, and if so exclude it
    imres - tuple for image resolution
    best_only - boolean to decide whether to return only the best image
    '''
    if detector is None:
        detector = cv2.FaceDetectorYN.create(
            "models/fd_yunet.onnx",
            "",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=.4
        )
    if type(img_name) is np.ndarray: # allow for two types of inputs
        img = img_name
        img_name = "cropped_face.png"
    # Initialize outputs
    faces = (0,[])
    if return_faces:
        cropped_imgs = []
    if img is None: # need to load image
        if not os.path.exists(img_name):
            print('[INFO] Image file does not exist, skipping')
            if return_faces:
                return detector,idx0,0,faces,cropped_imgs
            else:
                return detector, idx0
        fullname, file_extension = os.path.splitext(img_name)
        filename = os.path.basename(fullname)
        character_name = os.path.basename(os.path.dirname(filename))
        cropped_dir = os.path.join(cropped_dir,character_name)
        img = cv2.imread(img_name)
    else:
        filename = img_name
    if save_rect or save_square:
        os.makedirs(cropped_dir,exist_ok=True)
        if is_single_color(img): # basically one color, no features
            print("[INFO] Image is effectively single-color, skipping")
            detector = None # just in case something got wonky since it's uncharted territory
            if return_faces:
                return detector,idx0,0,faces,cropped_imgs
            else:
                return detector, idx0
    m,n,_ = img.shape
    detector.setInputSize((n,m))
    try:
        faces = detector.detect(img)
    except: # The image is simply too hot to handle, this is a rare bug
        print("[INFO] Image detect failed, skipping")
        detector = None # just in case something got wonky since it's uncharted territory
        if return_faces:
            return detector,idx0,0,faces,cropped_imgs
        else:
            return detector, idx0
    Nf = 0 # initialize in case the for loop is skipped
    npop = 0
    if faces[1] is not None:
        # First we need to check for and remove cases of boxes within boxes.
        # Setting a lower nms_threshold would remove the outer box, but if that larger box is around two smaller boxes,
        # it likely meaidxns the larger box is detecting features from two seperate but nearby heads
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
            faces = list(faces) # convert to list so we can assign values
            faces[1] = faces[1][keep_rect]
            faces = tuple(faces)
        # With the sorting process done, we need to remove any negative values.
        coords = np.maximum(0,coords)
        # There may still be values that are too large, but it's more convenient to sort this out within the loop
        Nf = len(coords)
        # Now loop through the faces that we have determined are indeed faces.
        if best_only:
            Nf = 1
        for idx in range(Nf):
            x1, y1, w, h = coords[idx,0:4]
            if min(w,h) < min_dim:
                faces = list(faces)
                faces[1] = np.delete(faces[1],idx-npop,0)
                faces = tuple(faces)
                npop += 1
                continue
            # First save cropped version as determined by Yunet.
            # This way we can compare the images with other cropped ones
            x2 = x1 + w
            if x2 > n:
                x2 = n
            y2 = y1 + h
            if y2 > m:
                y2 = m
            imgr = img[y1:y2, x1:x2, :]
            # we check here even if we are only saving square image, as if we DO want to save both we don't want to risk only saving one
            if is_single_color(imgr) and do_filtering:
                faces = list(faces)
                faces[1] = np.delete(faces[1],idx-npop,0)
                faces = tuple(faces)
                npop += 1
                continue
            if save_rect:
                namer =  os.path.join(cropped_dir,filename+'-rect'+str(idx0+idx)+'.png')
                cv2.imwrite(namer,imgr)
                if return_faces:
                    cropped_imgs.append(imgr)
            # Now get a square crop to use with CNN, which we can easily scale as needed.
            # Start by enlarging by 10% in all directions, to get full chin and ears, and more hair.
            if save_square:
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
                if imres[1] is not None: # need to resize image
                    imgs = cv2.resize(imgs,imres)
                names =  os.path.join(cropped_dir,filename+'-square'+str(idx0+idx)+'.png')
                cv2.imwrite(names,imgs)
                if return_faces:
                    cropped_imgs.append(imgs)
    if return_faces:
        return detector,idx0+Nf-npop,img,faces,cropped_imgs
    else:
        return detector,idx0+Nf-npop # so we don't have to re-initialize next time


def crop_objects(img_name="input_img.png", img=None, model=None, cropped_dir='Images/cropped-images', idx0=0, score_threshold=.3, min_dim=16, save_rect=True, 
               save_square=True, return_objs=False, do_filtering=True, imres=(None,None), best_only=False):
    '''
    INPUTS
    img_name - full path to input image if img is None, or output name if img is input image
    img - input image array
    detector - instance of cv2.FaceDetectorYN if we want to customize it
    cropped_dir - output directory for images, or parent directory of subdirectory for images if image_name comes from a directory for a character name
    idx0 - index of image within that image, can be used if multiple images with the same img_name will be used so we can iterate indx0 across function calls
    score_threshold - score (~probabaility) for a face deteciton to pass as valid
    min_dim - minimum dimension of a detected face, for it to be considered valid
    save_rect - boolean to determine whether or not to save the crop from cv2
    save_square - boolean to determine whether to save larger modified square crop
    return_objs - return extra variables to use, including list of cropped images
    do_filtering - check if object image is effectively single-color, and if so exclude it
    imres - tuple for image resolution
    best_only - boolean to decide whether to return only the best image
    '''
    if model is None:
        # Specify the path to model config and checkpoint file
        config_file = 'checkpoints/rtmdet_l_8xb32-300e_coco.py'
        checkpoint_file = 'checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
        # Build the model from a config file and a checkpoint file
        model = init_detector(config_file, checkpoint_file, device='cpu')
    if type(img_name) is not str: # allow for two types of inputs
        img = img_name
        img_name = "input_img.png"
    # Initialize outputs
    faces = (0,[])
    if return_objs:
        cropped_imgs = []
    if img is None: # need to load image
        if not os.path.exists(img_name):
            print('[INFO] Image file does not exist, skipping')
            if return_objs:
                return model,idx0,0,faces,cropped_imgs
            else:
                return model, idx0
        fullname, file_extension = os.path.splitext(img_name)
        filename = os.path.basename(fullname)
        character_name = os.path.basename(os.path.dirname(filename))
        cropped_dir = os.path.join(cropped_dir,character_name)
        img = mmcv.imread(img_name)
    else:
        filename = img_name
    if save_rect or save_square:
        os.makedirs(cropped_dir,exist_ok=True)
        if is_single_color(img): # basically one color, no features
            print("[INFO] Image is effectively single-color, skipping")
            model = None # just in case something got wonky since it's uncharted territory
            if return_objs:
                return model,idx0,0,faces,cropped_imgs
            else:
                return model, idx0
    m,n,_ = img.shape
    # Run inference
    result = inference_detector(model, img)
    # People are class 0, so remove those.
    mask = np.logical_and(result.pred_instances.scores>score_threshold,result.pred_instances.labels>0).bool()
    bboxes = np.array(result.pred_instances.bboxes[mask]).astype(np.int32) # N boxes with [x1,y1,x2,y2] coords
    Nb = len(bboxes)
    # Check for overlap
    if Nb > 1:
        areas = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
        max_overlap = .4 # max fraction of bbox that can intersect another box before we have a problem
        # Initialize variable to store ares and boolean mask array
        overlapped_areas = np.zeros((Nb,Nb))
        ovlp_wrt_js = np.zeros((Nb,Nb))
        keep_rect = np.full((Nb), True)
        for idx in range(Nb):
            for jdx in range(idx,Nb):
                if idx == jdx:
                    continue
                overlapped_areas[idx,jdx] = overlapped_area(bboxes[idx,:],bboxes[jdx,:])
                overlapped_areas[jdx,idx] = overlapped_areas[idx,jdx]
            ovlp_wrt_js[idx,:] = overlapped_areas[idx,:]/areas
            if sum(ovlp_wrt_js[idx,:]>max_overlap) > 1: # this box contains mutliple faces
                keep_rect[idx] = False
            elif sum(ovlp_wrt_js[idx,:]>max_overlap) == 1:
                keep_rect[ovlp_wrt_js[idx,:]>max_overlap] = False
        if not np.any(keep_rect):
            idxs = np.argwhere(ovlp_wrt_js == np.max(ovlp_wrt_js))
            keep_rect[idxs[0][0]] = True
        bboxes = bboxes[keep_rect,:]
    # Now loop through the faces that we have determined are indeed faces.
    Nb = len(bboxes)
    if best_only:
        Nb = min(Nb,1)
    idx = 0
    while idx < Nb:
        x1, y1, x2, y2 = bboxes[idx,:]
        w = x2 - x1
        h = y2 - y1
        if min(w,h) < min_dim:
            bboxes = np.delete(bboxes,idx,0)
            Nb -= 1
            continue
        # First save cropped version as determined by Yunet.
        # This way we can compare the images with other cropped ones
        if x2 > n:
            x2 = n
        if y2 > m:
            y2 = m
        imgr = img[y1:y2, x1:x2, :]
        # we check here even if we are only saving square image, as if we DO want to save both we don't want to risk only saving one
        if is_single_color(imgr) and do_filtering:
            bboxes = np.delete(bboxes,idx,0)
            Nb -= 1
            continue
        if save_rect:
            namer =  os.path.join(cropped_dir,filename+'-rect'+str(idx0+idx)+'.png')
            cv2.imwrite(namer,imgr)
            if return_objs:
                cropped_imgs.append(imgr)
        # Now get a square crop to use with CNN, which we can easily scale as needed.
        # Start by enlarging by 10% in all directions, to get full chin and ears, and more hair.
        if save_square:
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
            if imres[1] is not None: # need to resize image
                imgs = cv2.resize(imgs,imres)
            names =  os.path.join(cropped_dir,filename+'-square'+str(idx0+idx)+'.png')
            cv2.imwrite(names,imgs)
            if return_objs:
                cropped_imgs.append(imgs)
        idx += 1
    if return_objs:
        return model,idx0+Nb,img,cropped_imgs
    else:
        return model,idx0+Nb # so we don't have to re-initialize next time



def download_models():
    # Download face detection model
    link = 'https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx'
    r = requests.get(link)
    download_path = 'models'
    os.makedirs(download_path,exist_ok=True)
    file_name = 'fd_yunet.onnx'
    with open(os.path.join(download_path, file_name), 'wb') as fd:
        fd.write(r.content)
    # Download object detection model



def crop_video_frames(videos_dir,out_dir='Images/anime-frames-cropped',skip_frames=500,save_square=True,save_rect=False,imres=(96,96)):
    detector = None
    video_files = files_in_dir(videos_dir,["mkv","mp4","avi","webm"])
    for Ep,file in enumerate(video_files):
        Epstr = str(Ep+1)
        if len(Epstr) == 1:
            Epstr = '0' + Epstr
        print(f'[INFO] Extracting faces from {file}, designated as Ep{Epstr}')
        cap = cv2.VideoCapture(os.path.join(file))
        nfails = 0
        fidx = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                if fidx == skip_frames:
                    timestamp= cap.get(cv2.CAP_PROP_POS_MSEC)
                    hh = int(timestamp/(1000*3600))
                    rem_ms = (timestamp-3600*1000*hh)
                    mm = int(rem_ms/(1000*60))
                    rem_ms -= mm*60*1000
                    ss = int(rem_ms/1000)
                    rem_ms -= ss*1000
                    ms = int(rem_ms)
                    hh = str(hh)
                    if len(hh) == 1:
                        hh = '0' + hh
                    mm = str(mm)
                    if len(mm) == 1:
                        mm = '0' + mm
                    ss = str(ss)
                    if len(ss) == 1:
                        ss = '0' + ss
                    ms = str(ms)
                    ms = '0'*(3-len(ms)) + ms
                    cropped_name = 'Ep'+Epstr+'hh'+hh+'mm'+mm+'ss'+ss+'ms'+ms
                    detector,idx0 = crop_faces(img_name=cropped_name, img=frame, detector=detector, cropped_dir=out_dir,score_threshold=.9,save_rect=save_rect,save_square=save_square,imres=imres)
                    print(f'{idx0} images found for {cropped_name}')
                    fidx = 0
                else:
                    fidx += 1
            else:
                nfails += 1
                if nfails > 2:
                    cap.release()


def identical_images(img1,img2,L1_thresh=.05):
    # For this function we consider a lighter version of an image to be the same image,
    # as we want to know how many truly unique iamges we have.
    # Therefore, we start by normalizing.
    img1 = img1/np.max(img1)
    img2 = img2/np.max(img2)
    m1,n1,_ = img1.shape
    m2,n2,_ = img2.shape
    mindim = min(m1,n1,m2,n2)
    img1 = cv2.resize(img1, (mindim, mindim))
    img2 = cv2.resize(img2, (mindim, mindim))
    if img1.size != img2.size: # different dimensions
        return False
    L1 = np.sum(np.abs(img1-img2))/img1.size
    return L1 < L1_thresh


def remove_duplicates(images_dir):
    imgs_list = files_in_dir(images_dir)
    Li = len(imgs_list)
    is_duplicate = np.full(Li, False)
    for idx in range(Li):
        if is_duplicate[idx]:
            os.remove(imgs_list[idx])
            continue
        else:
            img1 = cv2.imread(imgs_list[idx])
        for jdx in range(idx+1,Li):
            if is_duplicate[jdx]:
                continue
            else:
                img2 = cv2.imread(imgs_list[jdx])
            is_duplicate[jdx] = identical_images(img1,img2)


def crop_orig_imgs(images_dir_top='Images/google-images-original', accepted_dir_top='Images/accepted-images', rejected_dir_top='Images/rejected-images',a_thresh=.7,r_thresh=.2):
    dirs = sorted(glob.glob(os.path.join(images_dir_top,'*')))
    recognizer = cv2.FaceRecognizerSF.create("models/fr_sface.onnx","")
    detector = None # initialize
    config = None
    for dir in dirs:
        # Get name of file from myanimelist that we can definitively trust to be the right character
        img1_name = glob.glob(os.path.join(dir.replace('google','myanimelist'),'*'))[0]
        character_name = os.path.basename(dir)
        # Make all directories possibly needed
        accepted_dir = os.path.join(accepted_dir_top,character_name)
        os.makedirs(accepted_dir,exist_ok=True)
        rejected_dir = os.path.join(rejected_dir_top,character_name)
        os.makedirs(rejected_dir,exist_ok=True)
        noface_dir = os.path.join('Images','google-images-noface',character_name)
        os.makedirs(noface_dir,exist_ok=True)
        # Now run detection on myanimelist image to get a baseline.
        detector, idx0, img1, faces1, cropped_faces1 = crop_faces(img_name=img1_name,detector=detector,cropped_dir=accepted_dir,save_rect=False,save_square=True,return_faces=True,do_filtering=False)
        face1_align = recognizer.alignCrop(img1, faces1[1][0])
        # Extract features
        face1_feature = recognizer.feature(face1_align)
        # Now get similar faces from google images
        imgs_list = files_in_dir(dir)
        crop_dir = os.path.join('Images','google-images-cropped',character_name)
        # Loop through google images for current character
        for file in imgs_list:
            _, idx0, img2, faces2, cropped_faces2 = crop_faces(img_name=file,detector=detector,cropped_dir=crop_dir,return_faces=True,score_threshold=.5)
            if idx0:
                for idx,face in enumerate(faces2[1]):
                    face2_align = recognizer.alignCrop(img2, face)
                    # Check colorspce of face.
                    accepted, rejected, config = is_in_colorpsace(face2_align,cfg_name='anime_colorspace_rect.cfg',config=config,za_thresh=1.5,zr_thresh=2)
                    # Extract features
                    face2_feature = recognizer.feature(face2_align)
                    # Get scores
                    cosine_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_COSINE)
                    l2_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_NORM_L2)
                    fullfile,ext = os.path.splitext(file)
                    out_name = os.path.basename(fullfile)+'_cos'+'%.3f'%(cosine_score)+ext
                    if accepted and cosine_score > a_thresh: # image passes, make sure to save square version
                        cv2.imwrite(os.path.join(accepted_dir,out_name),cropped_faces2[2*idx+1])
                    elif rejected and cosine_score < r_thresh: # image is definitely not the same person
                        cv2.imwrite(os.path.join(rejected_dir,out_name),cropped_faces2[2*idx+1])
            else:
                shutil.copy(file,os.path.join(noface_dir,os.path.basename(file)))


def get_not_this_anime(anime_file,imres=(96,96)):
    df = pd.read_csv(anime_file)
    anime_name = anime_file.split('-')[0]
    reject_strs = [anime_name]
    for idx in df.index:
        reject_strs.append(df.Name[idx])
        if type(df.Other_Names[idx]) == str:
            additional_keys = df.Other_Names[idx].split(',')
            reject_strs.extend(additional_keys)
    # Get dataset of anime images excluding our anime
    parallel_worker_threads(search_keys="anime characters",token_names="anime_characters",imgs_path="Images/other_anime-original",num_images=1000,
                            maxmissed=1000,reject_strs=reject_strs,simthresh=.3)
    imgs_list = files_in_dir("Images/other_anime-original")
    # This anime, Other anime, Manga, Not anime
    other_anime_data_dir = os.path.join("datasetsTOMN","other_anime")
    print(f"Generating dataset of other anime faces in {other_anime_data_dir}.")
    for file in imgs_list:
        crop_faces(img_name=file,cropped_dir=other_anime_data_dir,score_threshold=.9,save_rect=False,imres=imres)
    # Delete unneeded directories, and in doing so remove any potentially problematic images.
    shutil.rmtree("Images/other_anime-original")
    # Get dataset of irl faces that definitely won't be our anime
    parallel_worker_threads(search_keys="people's faces",token_names="people_faces",imgs_path="Images/not_anime-original",num_images=1000,simthresh=.2)
    imgs_list = files_in_dir("Images/not_anime-original")
    not_anime_data_dir = os.path.join("datasetsTON","not_anime")
    print(f"Generating dataset of non-anime faces in {not_anime_data_dir}.")
    for file in imgs_list:
        crop_faces(img_name=file,cropped_dir=not_anime_data_dir,score_threshold=.9,save_rect=False,imres=imres)
    # Delete unneeded directories, and in doing so remove any potentially problematic images.
    shutil.rmtree("Images/not_anime-original")


# This function takes as input a csv as generated by list_anime_characters,
# and then runs other functions to download google images for each character.
def get_character_images(anime_file,images_path='Images/character_images',imres=(96,96)):
    df = pd.read_csv(anime_file)
    anime_name = anime_file.split('-')[0]
    append_str = f" {anime_name} anime"
    for idx in df.index: # Loop through characters in our anime
        image_dir = os.path.join(images_path,df.Name[idx].replace(" ","_"))
        character_names = [df.Name[idx]]
        if type(df.Other_Names[idx]) == str: # Need to add character aliases
            additional_keys = df.Other_Names[idx].split(',')
            character_names.extend(additional_keys)
        search_keys = [s + append_str for s in character_names] # append anime name and "anime" in search text
        token_names = [s.replace(" ","_") for s in character_names]
        parallel_worker_threads(search_keys,token_names=token_names,imgs_path=image_dir,num_images=500,maxmissed=1000)
        imgs_list = files_in_dir(image_dir)
        print(f"Generating inital dataset of faces for {character_names[0]}.")
        character_dir = os.path.join('datasets_base',df.Name[idx].replace(" ","_"))
        for file in imgs_list:
            crop_faces(img_name=file,cropped_dir=character_dir,score_threshold=.4,save_rect=False,imres=imres)
    # Delete unneeded directories, and in doing so remove any potentially problematic images.
    shutil.rmtree(images_path)

def initialize_training_set(base_dir="Images/myanimelist-images-original",out_dir="datasets_training",imres=(96,96)):
    class_names = sorted(glob.glob(os.path.join(base_dir,"*")))
    for class_name in class_names:
        for file in os.listdir(class_name):
            crop_faces(os.path.join(class_name,file),cropped_dir=os.path.join(out_dir,os.path.basename(class_name)),score_threshold=.3,save_rect=False,imres=imres,best_only=True)

if __name__ == '__main__':
    # list_anime_characters('Monster','Images/myanimelist-images-original')
    # get_character_images("Monster-Characters.csv")
    # load_image('Images/google-images/Adolf_Junkers/Adolf_Junkers_0.webp')
    # remove_grayscale_images("Monster-Characters.csv",'Images/google-images')
    # check_gray('Images/google-images/Anna_Liebert/')
    # download_models()
    # image_name = 'Images/google-images-original/Robbie/Robbie_25.jpeg'
    # detector = crop_faces(image_name,cropped_dir='Images/google-images-cropped/Robbie')
    # crop_orig_imgs(a_thresh=.01)
    # crop_video_frames('Monster.S01.480p.NF.WEB-DL.DDP2.0.x264-Emmid',out_dir='datasetsTON/this_anime',skip_frames=400,save_rect=False,save_square=True)
    # crop_video_frames('/home/redxhat/Videos/Vinland_Saga',out_dir='Images/vinland-frames-cropped-rect',skip_frames=100,save_rect=True,save_square=False)

    # crop_video_frames('/home/redxhat/Videos/Vinland_Saga','Vinland-cropped')
    # remove_duplicates('Images/anime-frames-cropped')
    # get_colorspace('Images/anime-frames-cropped-rect',cfg_name='monster_colorspace.ini')
    # get_colorspace('Images/vinland-frames-cropped-rect',cfg_name='vinland_colorspace.ini')
    # filter_by_colorspace('Images/google-images-cropped/Adolf_Junkers/',cfg_name='monster_colorspace_rect.cfg',za_thresh=1)
    # filter_google_images()
    # create_unlabeled_set()
    # get_not_this_anime_("Monster-Characters.csv")
    crop_objects("Images/google-images-noface/Johan_Liebert/_Monster_307.jpeg")

