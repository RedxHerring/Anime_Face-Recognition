import glob
import os
import shutil
from utils import files_in_dir
from utils_ML import get_image_type

def filter_dataset_by_imagetype(dataset_dir_in="datasets_base",dataset_dir_out="datasets_anime",rejected_dir_top="Images/rejected_images"):
    dirs = glob.glob(os.path.join(dataset_dir_in,'*'))
    model = None
    for dir in dirs:
        images_list = files_in_dir(dir)
        accepted_dir = os.path.join(dataset_dir_out,os.path.basename(dir))
        os.makedirs(accepted_dir,exist_ok=True)
        rejected_dir = os.path.join(rejected_dir_top,os.path.basename(dir))
        os.makedirs(rejected_dir,exist_ok=True)
        for file in images_list:
            predicted_class_label, model = get_image_type(file,model=model)
            if predicted_class_label == "this_anime":
                shutil.copy(file,os.path.join(accepted_dir,os.path.basename(file)))
            else:
                name, ext = os.path.splitext(os.path.basename(file))
                out_name = os.path.join(rejected_dir,name+'-'+predicted_class_label+ext)
                shutil.copy(file,out_name)

if __name__ == "__main__":
    filter_dataset_by_imagetype("datasets_base")
