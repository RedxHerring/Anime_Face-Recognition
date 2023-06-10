import torch
# source /opt/intel/oneapi/setvars.sh
# export LD_PRELOAD=/usr/lib/libstdc++.so.6.0.31
# Define ML device globally in this file
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    try:
        import intel_extension_for_pytorch as ipex
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
from torchvision.models import efficientnet_v2_l
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



class AugmentedDataset(Dataset):
    def __init__(self, dataset, total_augmented_images, transform):
        self.dataset = dataset
        self.total_augmented_images = total_augmented_images
        self.transform = transform

    def __len__(self):
        return self.total_augmented_images

    def __getitem__(self, index):
        image, label = self.dataset[index % len(self.dataset)]
        image = self.transform(image)
        return image, label


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
    if augmentations_per_image == 1: # just one transform
        transform = transforms.Compose([transforms.RandomHorizontalFlip(1)]) # guaranteed flip
    else:
        transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0, 0.2), shear=0.2),
            transforms.RandomResizedCrop(imres, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    total_augmented_images = augmentations_per_image * len(image_files)
    print(f'Augmenting {int((1 - tvsplit) * total_augmented_images)} images into training and {int(tvsplit * total_augmented_images)} images into validation for {class_name}.')
    num_digits = int(np.ceil(np.log10(total_augmented_images)))
    Tvsplit = np.ceil(1/tvsplit)
    for imgf in image_files:
        img = Image.open(imgf)
        fullname, ext = os.path.splitext(imgf)
        basename = os.path.basename(fullname)
        if copy_base:
            shutil.copy(imgf,os.path.join(train_dir,basename+ext))
        for idx in range(augmentations_per_image):
            augmented_image = transform(img)
            if type(augmented_image) is torch.Tensor:
                augmented_image = transforms.ToPILImage()(augmented_image)
            idxstr = str(idx)
            idxstr = '_' + '0'*(num_digits-len(idxstr)) + idxstr
            if (idx+1)%Tvsplit:
                augmented_image.save(os.path.join(train_dir, basename + idxstr + ext))
            else:
                augmented_image.save(os.path.join(val_dir, basename + idxstr + ext))


def augment_dataset(in_dir='dataset_base', out_dir='dataset_augmented', num_augmented=100, tvsplit=.1, copy_base=False):
    '''
    Loop through subdirectories and augment datasets.
    '''
    shutil.rmtree(os.path.join(in_dir,'train'),ignore_errors=True)
    shutil.rmtree(os.path.join(in_dir,'val'),ignore_errors=True)
    class_dirs = sorted(os.listdir(in_dir))
    for dir in class_dirs:
        augment_images(os.path.join(in_dir,dir),os.path.join(out_dir,dir),total_images=num_augmented,tvsplit=tvsplit, copy_base=copy_base)


def train_face_recognition_tf(training_dir='datasets_training', validation_dir='datasets_anime', imres=(96, 96), out_name='models/saved_model.h5', 
                            batch_size=16, reg=.01, drprate=.7, num_epochs=5, lr=.001, model=None):
    warnings.filterwarnings("ignore")
    checkpoint_path = "checkpt"
    create_missing_classes(training_dir=training_dir, validation_dir=validation_dir)

    # Prepare training and validation datasets
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    training_set = build_dataset(training_dir, imres, "training")
    class_names = tuple(training_set.class_names)
    training_size = training_set.cardinality().numpy()
    training_set = training_set.unbatch().repeat().batch(batch_size)
    training_set = training_set.map(lambda images, labels: (normalization_layer(images), labels))
    
    validation_set = build_dataset(validation_dir, subset='validation')
    validation_size = validation_set.cardinality().numpy()
    validation_set = validation_set.unbatch().batch(batch_size)
    validation_set = validation_set.map(lambda images, labels: (normalization_layer(images), labels))

    do_fine_tuning = True
    if model is None:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=imres + (3,)),
            hub.KerasLayer('https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1', trainable=do_fine_tuning),
            tf.keras.layers.Dropout(rate=drprate),
            tf.keras.layers.Dense(len(class_names),
                                kernel_regularizer=tf.keras.regularizers.l2(reg))
        ])
    else:
        # Update the l2 regularization of the input model
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_regularizer = tf.keras.regularizers.l2(reg)
    decay_rate = lr / num_epochs
    model.compile(
        optimizer=tf.keras.optimizers.legacy.SGD(
            learning_rate=lr, momentum=0.9, nesterov=False, decay=decay_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(name='top-5-accuracy')],
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    steps_per_epoch = training_size // batch_size
    validation_steps = validation_size // batch_size

    hist = model.fit(
        training_set,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_set,
        validation_steps=validation_steps,
        callbacks=[reduce_lr, checkpoint_callback, callback]
    ).history
    model.save(out_name)
    return model, hist


def train_face_recognition_torch(recursive_dir='datasets_recursive', source_dir='datasets_anime', imres=(96, 96), out_name='models/saved_model.pt'):
    batch_size = 16
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import intel_extension_for_pytorch as ipex
            device = 'xpu'
        except: 
            device = "cpu"
    
    create_missing_classes(training_dir=recursive_dir, validation_dir=source_dir)

    # Data transformations
    data_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(imres, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load datasets
    train_dataset = datasets.ImageFolder(root=recursive_dir, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_dataset = datasets.ImageFolder(root=source_dir, transform=data_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Define the model
    model = efficientnet_v2_l()
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # intitial train
    model.train()
    ### set to device and continue
    model.to(device)
    criterion.to(device)
    if device == 'xpu':
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {val_loss / len(validation_loader):.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%")
    # Save the trained model
    torch.save(model.state_dict(), out_name)
    return model


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

    if model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

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
        model_ft = models.efficientnet_b7(weights='DEFAULT')
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 600

    elif model_name == "efficientnetv2":
        """ EfficientnetV2 L
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.efficientnet_v2_l(weights='DEFAULT')
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 600

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, device='cpu'):
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


def run_finetune_model_torch(train_dir='datasets_iterative0',val_dir='datasets_anime',model_name="resnet",out_name="models/saved_model.pt"):
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    # Number of classes in the dataset
    num_classes = len(os.listdir(train_dir))
    # Batch size for training (change depending on how much memory you have)
    batch_size = 8
    # Number of epochs to train for
    num_epochs = 15
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # Print the model we just instantiated
    print(model_ft)
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
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
    # Detect if we have a GPU available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import intel_extension_for_pytorch as ipex
            device = 'xpu'
        except: 
            device = "cpu"
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
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"),device=device)
    torch.save(model_ft.state_dict(), out_name)
    return model_ft


def get_image_scores_torch(model=None, model_path="models/saved_model.pt"):
    if model is None:
        model = torch.load(model_path)


class OneShotDataset(Dataset):
    def __init__(self, dataset, num_augmented_images, normalization_transform, augmentation_transform):
        self.dataset = dataset
        self.num_augmented_images = num_augmented_images
        self.augmentation_transform = augmentation_transform
        self.normalization_transform = normalization_transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        augmented_images = []
        augmented_labels = []
        image_pil = transforms.ToPILImage()(image)  # Convert tensor to PIL Image
        for _ in range(self.num_augmented_images):
            augmented_image = self.augmentation_transform(image_pil)
            augmented_images.append(self.normalization_transform(transforms.ToPILImage()(augmented_image)))
            augmented_labels.append(label)
        return self.normalization_transform(transforms.ToPILImage()(image)), torch.tensor(label), augmented_images, torch.tensor(augmented_labels)

    def __len__(self):
        return len(self.dataset)


def run_face_recognition_1shot_torch(training_dir='datasets_training', validation_dir='datasets_anime', 
                                      imres=(96, 96), num_augmented_images=100, out_name='models/saved_model.pt', 
                                      reg=0.01, model=None, model_name='inception', feature_extract=False):
    batch_size = 16
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import intel_extension_for_pytorch as ipex
            device = 'xpu'
        except: 
            device = "cpu"

    create_missing_classes(training_dir=training_dir, validation_dir=validation_dir)
    
    transform = transforms.Compose([
        transforms.Resize(imres),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(imres, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    training_set = ImageFolder(training_dir, transform=transform)
    class_names = training_set.classes

    training_set = OneShotDataset(training_set, num_augmented_images, transform, transform)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    validation_set = ImageFolder(validation_dir, transform=transforms.Compose([
        transforms.Resize(imres),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dataloaders_dict = {'train':training_loader, 'val':validation_loader}

    if model is None:
        # Initialize the model for this run
        model, input_size = initialize_model(model_name, len(class_names), feature_extract, use_pretrained=True)
        # Print the model we just instantiated
        print(model)
    model.to(device)
    
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9, weight_decay=reg)

    if device == 'xpu':
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    
    num_epochs = 10

    # Train and evaluate
    model, hist = train_model(model, dataloaders_dict, criterion, optimizer, device, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    model.load_state_dict(torch.load(out_name))
    return model


def train_face_recognition_1shot_torch(training_dir='datasets_training', validation_dir='datasets_anime', 
                                      imres=(96, 96), num_augmented_images=100, out_name='models/saved_model.pt', 
                                      reg=0.01, model=None):
    batch_size = 16
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import intel_extension_for_pytorch as ipex
            device = 'xpu'
        except: 
            device = "cpu"

    create_missing_classes(training_dir=training_dir, validation_dir=validation_dir)
    
    transform = transforms.Compose([
        transforms.Resize(imres),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(imres, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    training_set = ImageFolder(training_dir, transform=transform)
    class_names = training_set.classes

    training_set = OneShotDataset(training_set, num_augmented_images, transform, transform)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    validation_set = ImageFolder(validation_dir, transform=transforms.Compose([
        transforms.Resize(imres),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if model is None:
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.AuxLogits.fc.in_features
        num_classes = len(class_names)
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=reg)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=0.00001)

    if device == 'xpu':
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    
    num_epochs = 10
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for images, labels, augmented_images, augmented_labels in training_loader:
            images, labels = images.to(device), labels.to(device)
            augmented_images = [img.to(device) for img in augmented_images]
            augmented_labels = [lbl.to(device) for lbl in augmented_labels]

            optimizer.zero_grad()
            outputs = model(torch.cat([images] + augmented_images, dim=0))
            labels = torch.cat([labels] + augmented_labels, dim=0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(training_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(validation_loader.dataset)
        val_acc = correct / total

        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')

        # Early stopping based on validation loss
        if epoch > 4 and val_loss < best_val_loss:
            early_stop_counter += 1
            if early_stop_counter >= 3:
                print('Early stopping triggered. Training stopped.')
                break
        else:
            early_stop_counter = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_name)

    model.load_state_dict(torch.load(out_name))
    return model


def load_existing_model(model_name='models/saved_model.h5'):
    model = tf.keras.models.load_model(model_name,
       custom_objects={'KerasLayer':hub.KerasLayer})
    return model


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
    augment_dataset('datasetsTOMON','datasetsTOMON',4000,tvsplit=.1)
    # train_val_split('datasetsTOMON/')
    # run_finetune_model_torch(train_dir='datasetsTOMON/train',val_dir='datasetsTOMON/val',model_name="resnet101",out_name="models/image_model.pt")
    # run_finetune_model_torch(train_dir='datasets_augmented1tv/train',val_dir='datasets_augmented1tv/val',model_name="resnet",out_name="models/FRmodel1.pt")
    # model = train_face_recognition_1shot_torch(training_dir='datasets_iterative0',validation_dir='datasets_anime',imres=(96,96),num_augmented_images=150,out_name='models/one_shot.pt')

