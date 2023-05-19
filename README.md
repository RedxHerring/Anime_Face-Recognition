# Monster_Face-Recognition
Recognizing and grouping faces from Monster, the anime.

Image downloading based on https://github.com/ohyicong/Google-Image-Scraper


480p source from https://nyaa.si/view/1611098

To run this code, first run
conda env create -f environment.yml
If new modules are installed, update the yaml with
conda env export > environment.yml

Download models from https://drive.google.com/drive/folders/14kSKdc4b3xKzNqow7bELE9Qqby3h3iTE?usp=share_link
Place them in the models/ directory

To git push and pull, example for ssh given below:
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/hpowell-id_ed25519