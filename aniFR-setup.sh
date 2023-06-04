# run this to create conda environment and install special dependencies within

conda env create -f environment.yml
conda activate aniFR
mim download mmdet --config rtmdet_l_8xb32-300e_coco --dest ./checkpoints
