import torch
# source /opt/intel/oneapi/setvars.sh
# export LD_PRELOAD=/usr/lib/libstdc++.so.6.0.31
# Define ML device globally in this file
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    try:
        import intel_extension_for_pytorch as ipex
        model = torch.nn.Conv2d(3, 4, (16,16)) # create dummy model
        model.eval()
        model = model.to('xpu')
        # If we get this far without error, this should be ok
        device = 'xpu'
    except: 
        device = "cpu"

import numpy as np
import glob
import os
import time
import copy
import warnings
import random
import shutil
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import shutil

from utils import files_in_dir, remove_duplicates

import cv2
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_val_split(train_dir_top, tvsplit=0.1):
    classes = list(set(os.listdir(train_dir_top)) - set(['train', 'val']))
    shutil.rmtree(os.path.join(train_dir_top,'train'),ignore_errors=True)
    shutil.rmtree(os.path.join(train_dir_top,'val'),ignore_errors=True)
    for img_class in classes:
        imgs_list = files_in_dir(os.path.join(train_dir_top,img_class))
        train_dir = os.path.join(train_dir_top,'train',img_class)
        val_dir = os.path.join(train_dir_top,'val',img_class)
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        for file in imgs_list:
            if np.random.rand() > tvsplit:
                shutil.copy(file,os.path.join(train_dir,os.path.basename(file)))
            else:
                shutil.copy(file,os.path.join(val_dir,os.path.basename(file)))


def create_missing_classes(training_dir,validation_dir):
    training_classes = set(os.listdir(training_dir))
    validation_classes = set(os.listdir(validation_dir))
    missing_classes = training_classes - validation_classes
    for missing_class in missing_classes:
        out_dir = os.path.join(validation_dir, missing_class)
        os.makedirs(out_dir)
        images_list = files_in_dir(os.path.join(training_dir,missing_class))
        img_name = images_list[0]
        img = cv2.flip(cv2.imread(img_name), 1) # load and flip horizontally
        base_name, ext = os.path.splitext(os.path.basename(img_name))
        out_name = os.path.join(out_dir,base_name+'_hflip'+ext)
        cv2.imwrite(out_name,img)


def augment_images(in_dir, out_dir, total_images=50, imres=(96, 96), tvsplit=0.2, copy_base=False):
    '''
    Given a single directory with images and a desired number of total images, 
    augment the images as needed to meet or excceed the desired number
    INPUTS
    in_dir - single directory with images inside, treated as unique base images
    out_dir - directory for augmented images, which may not contain any images from in_dir
    total_images - total nuumber of iamges to be saved in out_dir
    imres - tuple for output resolution
    tvsplit - fraction of images to be saved in validation directory
    copy_base - boolean to copy over images used for augmentation into the output training set
    '''
    out_dir_top = os.path.dirname(out_dir)
    class_name = os.path.basename(out_dir)
    train_dir = os.path.join(out_dir_top, 'train', class_name)
    val_dir = os.path.join(out_dir_top, 'val', class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    # Get image files
    image_files = files_in_dir(in_dir)
    if copy_base:
        num_augmented = max(0, total_images-len(image_files))
    else:
        num_augmented = total_images
    augmentations_per_image = int(np.ceil(num_augmented / len(image_files)))
    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0, 0.2), shear=0.2),
        transforms.RandomResizedCrop(imres, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(p=.25),
        transforms.ToTensor(),
    ])
    total_augmented_images = augmentations_per_image * len(image_files)
    print(f'Augmenting {int((1 - tvsplit) * total_augmented_images)} images into training and {int(tvsplit * total_augmented_images)} images into validation for {class_name}.')
    if total_augmented_images:
        num_digits = int(np.ceil(np.log10(total_augmented_images)))
    else:
        num_digits = 1
    for imgf in image_files:
        img = Image.open(imgf)
        fullname, ext = os.path.splitext(imgf)
        basename = os.path.basename(fullname)
        if copy_base:
            if np.random.rand() > tvsplit:
                shutil.copy(imgf,os.path.join(train_dir,basename+ext))
            else:
                shutil.copy(imgf,os.path.join(val_dir,basename+ext))
        for idx in range(augmentations_per_image):
            augmented_image = transform(img)
            if type(augmented_image) is torch.Tensor:
                augmented_image = transforms.ToPILImage()(augmented_image)
            idxstr = str(idx)
            idxstr = '_' + '0'*(num_digits-len(idxstr)) + idxstr
            if np.random.rand() > tvsplit:
                augmented_image.save(os.path.join(train_dir, basename + idxstr + ext))
            else:
                augmented_image.save(os.path.join(val_dir, basename + idxstr + ext))


def augment_dataset(in_dir='dataset_base', out_dir='dataset_augmented', num_augmented=100, tvsplit=.1, imres=(96,96), copy_base=False):
    '''
    Loop through subdirectories and augment datasets.
    '''
    # reset tv split
    shutil.rmtree(os.path.join(out_dir,'train'),ignore_errors=True)
    shutil.rmtree(os.path.join(out_dir,'val'),ignore_errors=True)
    class_dirs = sorted(os.listdir(in_dir))
    for dir in class_dirs:
        augment_images(os.path.join(in_dir,dir),os.path.join(out_dir,dir),total_images=num_augmented,imres=imres,tvsplit=tvsplit,copy_base=copy_base)


# helper functions from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        ft_transforms = models.ResNet50_Weights.IMAGENET1K_V2.transforms 

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        ft_transforms = models.ResNet152_Weights.IMAGENET1K_V2.transforms 

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    
    elif model_name == "efficientnet":
        """ Efficientnet B7
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        if use_pretrained:
            model_ft = models.efficientnet_b7(weights='DEFAULT')
        else:
            model_ft = models.efficientnet_b7()
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = torch.nn.Linear(model_ft.classifier.in_features, num_classes)
        input_size = 600

    elif model_name == "efficientnetv2S":
        """ EfficientnetV2 S
        """
        if use_pretrained:
            model_ft = models.efficientnet_v2_s(weights='DEFAULT')
        else:
            model_ft = models.efficientnet_v2_s()
        set_parameter_requires_grad(model_ft, feature_extract)
        # Get the number of output features from the last layer
        num_features = model_ft.features[-1][1].num_features
        # Replace the last layer with a new linear layer with num_classes outputs
        model_ft.classifier = torch.nn.Linear(num_features, num_classes)
        input_size = 384
        ft_transforms = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms
    
    elif model_name == "efficientnetv2L":
        """ EfficientnetV2 L
        """
        if use_pretrained:
            model_ft = models.efficientnet_v2_l(weights='DEFAULT')
        else:
            model_ft = models.efficientnet_v2_l()
        set_parameter_requires_grad(model_ft, feature_extract)
        # Get the number of output features from the last layer
        num_features = model_ft.features[-1][1].num_features
        # Replace the last layer with a new linear layer with num_classes outputs
        model_ft.classifier = torch.nn.Linear(num_features, num_classes)
        input_size = 400
        # input_size = 480

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size, ft_transforms


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, l1_lambda = 0.001):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    # L1 regularization
                    if l1_lambda > 0:
                        l1_reg = torch.tensor(0.).to(device)
                        for name, param in model.named_parameters():
                            if 'bias' not in name:
                                l1_reg += torch.norm(param, p=1)
                        loss += l1_lambda * l1_reg
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print(f'{phase} - Loss: {epoch_loss}, Acc: {running_corrects}/{len(dataloaders[phase].dataset)}')
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def run_finetune_model_torch(train_dir='datasets_iterative0', val_dir='datasets_anime', model_name="resnet", feature_extract=False, l1_lambda=0.001, 
                                out_name="models/saved_model.pth", batch_size=32, num_epochs=10, use_pretrained=True, learning_rate=.001):
    '''
    INPUTS
    train_dir - 
    val_dir - 
    model_name - general model type to feed into initialize_model()
    feature_extract - Flag for feature extracting. When False, we finetune the whole model. When True, we only update the reshaped layer params.
    l1_lambda - for L1 regularization
    out_name - saved name for model.state_dict()
    batch_size - Batch size for training (change depending on how much memory you have)
    num_epochs - Number of epochs to train for
    use_pretrained - set to false if we want to laod in model without pretrained weights
    '''
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    # Number of classes in the dataset
    num_classes = len(os.listdir(train_dir))
    # Initialize the model for this run
    model_ft, input_size, ft_transforms = initialize_model(model_name, num_classes, feature_extract, use_pretrained=use_pretrained)
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(ft_transforms.keywords['resize_size']),
            transforms.CenterCrop(ft_transforms.keywords['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(ft_transforms.keywords['resize_size']),
            transforms.CenterCrop(ft_transforms.keywords['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {'train': datasets.ImageFolder(train_dir, data_transforms['train']),
                        'val': datasets.ImageFolder(val_dir, data_transforms['val'])}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    # Send the model to GPU
    model_ft = model_ft.to(device)
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    if device == 'xpu':
        model_ft, optimizer_ft = ipex.optimize(model_ft,optimizer=optimizer_ft)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"), l1_lambda=l1_lambda)
    torch.save(model_ft.state_dict(), out_name)
    # return trained model and ordered classes
    return model_ft, image_datasets['train'].classes


def classify_single_image(image_path, class_names, model=None, model_path="models/saved_model.pt", imres=(96,96)):
    if model is None:
        model = torch.load(model_path)
    model.to(device)
    model.eval()
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize(imres),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization values
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        score, predicted_idx = torch.max(outputs, 1)
        predicted_label = class_names[predicted_idx.item()]
    return predicted_label, score

def get_image_scores_torch(model=None, model_path="models/saved_model.pt"):
    if model is None:
        model = torch.load(model_path)


def get_img_scores_tf(img_name, imres=(96,96), model=None):
    warnings.filterwarnings("ignore")
    if model is None:
        model = load_existing_model()
    img = tf.keras.utils.load_img(
        img_name , target_size=imres
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    img_array = normalization_layer(img_array)
    image = img_array[0,:,:,:]
    predictions = model.predict(img_array,verbose=0)
    return tf.nn.softmax(predictions[0])


def classify_character_tf(character, class_names, imres=(96, 96), source_dir='datasets_anime', out_dir='datasets_recursive', model=None, best_only=False, ac_min=.1,ac_max=.6,nmatches_thresh=50):
    '''
    INPUTS
    nmatches_thresh - if we get more matches than this we should move our cutoff threshold up significantly
    '''
    img_files = files_in_dir(os.path.join(source_dir,character)) # get full filenames
    Nf = len(img_files)
    accuracies = np.zeros(Nf)
    for fidx,img_f in enumerate(img_files): # Loop through files
        scores = get_img_scores_tf(img_f,imres,model)
        if scores is not None:
            predicted_index = np.argsort(scores)[-5:][::-1]
            if best_only: # only consider the image if it's more likely to be this class than any other
                cidx = predicted_index[0]
                if class_names[cidx] == character:
                    accuracies[fidx] = scores[cidx]
            else: # consider the image as long as one of its more likely classes is the right one
                for cidx in predicted_index: # Loop through top 5 most likely classes
                    if class_names[cidx] == character:
                        accuracies[fidx] = scores[cidx]
    histogram, edges = np.histogram(accuracies)
    # finding a decent threshold by looking for local minima and choosing the rightmost one
    local_minima = find_peaks(np.negative(histogram))
    if local_minima[0].size != 0:
        cutoff = min(ac_max,max(ac_min,edges[local_minima[0][-1]]))
        is_match = accuracies >= cutoff
        if sum(is_match) > nmatches_thresh: # so many matches that we need to be more selective
            num_accept = (sum(is_match)+2*nmatches_thresh)//3 # weighted average that takes into account this class
            cutoff = (cutoff + 2*np.sort(accuracies)[-num_accept])/3 # weighted average to create new cutoff
            is_match = accuracies >= cutoff
        matches = np.flatnonzero(is_match)
        # plt.stairs(histogram, edges, fill=True)
        # plt.show()
        for match in matches:
            file2 = os.path.join(out_dir,character,os.path.basename(img_files[match]))
            print(f'Copying {img_files[match]} to {file2}')
            shutil.copy(img_files[match],file2)
        print(f"Cutoff of {cutoff} score led to {len(matches)}/{Nf} = {len(matches)/Nf}")
        return (len(matches)/Nf)
    return 0


def classify_all_characters_tf(in_dir='datasets_anime', out_dir='datasets_recursive', model_name='models/saved_model.h5', best_only=False, ac_max=.5,ac_min=.1):
    training_set = build_dataset(base_dir=in_dir)
    class_names = tuple(training_set.class_names)
    characters = sorted(os.listdir(in_dir))
    C = len(characters)
    overall_matches = np.zeros(C)
    model = load_existing_model(model_name)
    for idx,character in enumerate(characters):
        print(f'{idx+1}/{C}: Character: {character}')
        detection_rate = classify_character_tf(character, class_names,source_dir=in_dir,out_dir=out_dir,model=model,best_only=best_only,ac_min=ac_min,ac_max=ac_max)
        # Check for duplicates.
        remove_duplicates(os.path.join(out_dir,character))
        overall_matches[idx] = detection_rate
    print(overall_matches)
    print(f'Overall detection rate: {np.mean(overall_matches)}')


def classify_all_images_tf(in_dir='datasets_anime', out_dir='datasets_recursive', model_name='models/saved_model.h5', ac_thresh=.5, imres=(96,96), 
                            nmatches_thresh=100, use_hists=True):
    '''
    INPUTS
    in_dir - directory containing folders with images, or just images with no folders
    out_dir - directory with a folder for each class, assumed to be built in advance
    model_name - tensorflow trained model to use to classification
    a_thresh - score to accept image as valid
    imrs - tuple storing resolution of images used
    '''
    training_set = build_dataset(base_dir=out_dir)
    class_names = tuple(training_set.class_names)
    C = len(class_names)
    overall_matches = np.zeros(C)
    model = load_existing_model(model_name)
    image_dirs = next(os.walk(in_dir))[1]
    if len(image_dirs):
        imgs_list = []
        for image_dir in image_dirs:
            imgs_list.extend(files_in_dir(os.path.join(in_dir,image_dir)))
    else:
        imgs_list = files_in_dir(in_dir)
    # Now either sort files globally or decide individually
    if use_hists:
        Nf = len(imgs_list)
        # Initialize matrix to store all scores for all files and all classes
        scores_mat = np.zeros((Nf,C))
        # Initialize matrix to store class index in 0th column and best score in 1st
        scores_CS = np.zeros((Nf,2))
        for idx, img_file in enumerate(imgs_list):
            scores = get_img_scores_tf(img_file,imres,model)
            if scores is not None:
                scores_mat[idx,:] = scores
                scores_CS[idx,0] = np.argmax(scores)
                scores_CS[idx,1] = max(scores)
        # Having looped through all files, we can decide how well a file fits a given class globally
        for idx in range(C):
            is_class = scores_CS[:,0]==idx
            if not np.any(is_class):
                continue
            accuracies = scores_CS[is_class,1] # get scores for each image that has this class as its best
            histogram, edges = np.histogram(accuracies)
            # finding a decent threshold by looking for local minima and choosing the rightmost one
            local_minima = find_peaks(np.negative(histogram))
            if local_minima[0].size:
                cutoff = max(ac_thresh,edges[local_minima[0][-1]])
                is_match = accuracies >= cutoff
                if sum(is_match) > nmatches_thresh: # so many matches that we need to be more selective
                    num_accept = (sum(is_match)+2*nmatches_thresh)//3 # weighted average that takes into account this class
                    cutoff = (cutoff + 2*np.sort(accuracies)[-num_accept])/3 # weighted average to create new cutoff
                    is_match = accuracies >= cutoff
                matches = np.flatnonzero(is_class)[np.flatnonzero(is_match)]
                for match in matches:
                    file2 = os.path.join(out_dir,class_names[idx],os.path.basename(imgs_list[match]))
                    print(f'Copying {imgs_list[match]} to {file2}')
                    shutil.copy(imgs_list[match],file2)
                print(f"Class {idx} - {class_names[idx]}: Cutoff of {cutoff} score led to {len(matches)}/{sum(is_class)} = {len(matches)/sum(is_class)}")
    else:
        for img_file in imgs_list:
            score = get_img_scores_tf(img_file,imres,model)
            if score is not None:
                if max(score) > ac_thresh:
                    cidx = np.argmax(score)
                    print(f'{img_file} found to be member of {class_names[cidx]} with score {max(score)}.')
                    shutil.copy(img_file,os.path.join(out_dir,class_names[cidx],os.path.basename(img_file)))
    
    for character in os.listdir(out_dir):
        # Check for duplicates.
        remove_duplicates(os.path.join(out_dir,character))




if __name__ == "__main__":
    # classify_image_type()
    # model, hist = train_face_recognition_tf()
    # model = load_existing_model()
    # augment_images('datasets_iterative0/Kenzou_Tenma','augmented_images',150)
    # classify_all_characters_tf('datasets_anime','datasets_iterative1',model_name='models/FRmodel2.h5',ac_min=.15,ac_max=.2)
    # augment_dataset('datasetsTOMON','datasetsTOMON',5000,tvsplit=.1,imres=(256,256),copy_base=True)
    # train_val_split('datasetsTOMON/')
    model_name = "efficientnetv2S"
    feature_extract=True
    model_ft, classnames = run_finetune_model_torch(train_dir='datasetsTOMON/train',val_dir='datasetsTOMON/val',model_name=model_name,l1_lambda=.00001, 
                                                    batch_size=16, out_name="models/image_model.pth",feature_extract=feature_extract,use_pretrained=False,
                                                    learning_rate=.004,num_epochs=20)
    # classnames = ['faces', 'manga', 'not_anime', 'objects', 'other_anime']
    _, input_size = initialize_model(model_name, len(classnames), feature_extract, use_pretrained=True)
    image_path = "datasetsTOMON/faces/Ep62hh00mm03ss03ms933-square0.png"
    predicted_label, score = classify_single_image(image_path,classnames,model=model_ft,model_path="models/image_model.pt",
                                                    imres=(input_size,input_size))
    print(f"Predicted Label for {image_path}: {predicted_label} with score {score}.")
    # run_finetune_model_torch(train_dir='datasets_augmented1tv/train',val_dir='datasets_augmented1tv/val',model_name="resnet",out_name="models/FRmodel1.pt")
    # model = train_face_recognition_1shot_torch(training_dir='datasets_iterative0',validation_dir='datasets_anime',imres=(96,96),num_augmented_images=150,out_name='models/one_shot.pt')

'''
resnet50
Epoch 9/9
----------
train - Loss: 27.603197657214416, Acc: 26108/28397
val - Loss: 27.57435869542977, Acc: 2927/3153

resnet152
Epoch 9/9
----------
train - Loss: 46.162073462279025, Acc: 26263/28397
val - Loss: 46.13337387631819, Acc: 2944/3153
'''