# run this to create conda environment and install special dependencies within

conda create -n aniFR python=3.10.10 -y
conda activate aniFR
conda install pytorch torchvision -c pytorch -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
mim install mmpose

# For verification
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python testmm.py