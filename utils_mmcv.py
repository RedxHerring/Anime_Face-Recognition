from utils import crop_faces, files_in_dir, remove_colored_images
from runGoogleImagScraper import parallel_worker_threads
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

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
    mask = np.logical_and(np.logical_and(result.pred_instances.scores>score_threshold,result.pred_instances.labels!=0),result.pred_instances.labels!=4).bool()
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


def get_not_this_anime(anime_file, dataset_top='datasetsTOMON', imres=(96,96)):
    df = pd.read_csv(anime_file)
    anime_name = anime_file.split('-')[0]
    # Get dataset of anime images excluding our anime
    reject_strs = [anime_name]
    for idx in df.index:
        reject_strs.append(df.Name[idx])
        if type(df.Other_Names[idx]) == str:
            additional_keys = df.Other_Names[idx].split(',')
            reject_strs.extend(additional_keys)
    anime_names = ['One Piece', 'Naruto', 'Jojos', 'Bleach', 'Monster', 'Gintama']
    other_anime = list(set(anime_names)-set([anime_name]))
    other_anime_queries = [s + " anime characters" for s in other_anime] 
    other_anime_queries.append("other anime")
    token_names = [s.replace(" ","_") for s in other_anime_queries]
    parallel_worker_threads(search_keys=other_anime_queries,token_names=token_names,imgs_path="Images/other_anime-original",num_images=500,
                            maxmissed=1000,reject_strs=reject_strs,simthresh=.3)
    imgs_list = files_in_dir("Images/other_anime-original")
    # This anime, Other anime, Manga, Objects, Not anime
    out_data_dir = os.path.join(dataset_top,"other_anime")
    print(f"Generating dataset of other anime faces in {out_data_dir}.")
    for file in imgs_list:
        crop_faces(img_name=file,cropped_dir=out_data_dir,score_threshold=.9,save_rect=False,imres=imres)
    # Delete unneeded directories, and in doing so remove any potentially problematic images.
    shutil.rmtree("Images/other_anime-original")
    
    # Get dataset of manga images for the anime.
    search_query = f"{anime_name} manga characters"
    parallel_worker_threads(search_keys=search_query,token_names="manga_characters",imgs_path="Images/manga-original",num_images=1000,
                            maxmissed=1000,reject_strs=reject_strs,simthresh=.3)
    out_data_dir = os.path.join(dataset_top,"manga")
    remove_colored_images("Images/manga-original") # keep only black-and-white downloaded images
    imgs_list = files_in_dir("Images/manga-original")
    print(f"Generating dataset of {anime_name} manga faces in {out_data_dir}.")
    for file in imgs_list:
        crop_faces(img_name=file,cropped_dir=out_data_dir,score_threshold=.5,save_rect=False,imres=imres)
    # Delete unneeded directories, and in doing so remove any potentially problematic images.
    shutil.rmtree("Images/manga-original")

    # Get dataset of irl faces that definitely won't be our anime
    parallel_worker_threads(search_keys="people's faces",token_names="people_faces",imgs_path="Images/not_anime-original",num_images=1000,simthresh=.2)
    imgs_list = files_in_dir("Images/not_anime-original")
    out_data_dir = os.path.join(dataset_top,"not_anime")
    print(f"Generating dataset of non-anime faces in {out_data_dir}.")
    for file in imgs_list:
        crop_faces(img_name=file,cropped_dir=out_data_dir,score_threshold=.9,save_rect=False,imres=imres)
    # Delete unneeded directories, and in doing so remove any potentially problematic images.
    shutil.rmtree("Images/not_anime-original")


if __name__ == "__main__":
    get_not_this_anime("monster-Characters.csv",imres=(256,256))