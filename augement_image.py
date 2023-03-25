# This python script will take as input a single image file name and an output directory
# It will turn the one image in the input into several iamges in the output directory
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

file_name = sys.argv[1]
output_dir = sys.argv[2]
if len(sys.argv) > 3:
    do_plots = sys.argv[3]
else:
    do_plots = False

# Using https://towardsdatascience.com/image-augmentation-examples-in-python-d552c26f2873

# Set inital image matrix
img = np.array(Image.open(file_name))
if do_plots:
    plt.imshow(img)
    plt.show()

# Flipping images with Numpy
flipped_img = np.fliplr(img)
if do_plots:
    plt.imshow(flipped_img)
    plt.show()


