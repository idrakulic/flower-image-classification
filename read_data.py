
from torchvision import transforms, datasets, models
import torch
from PIL import Image
import json

def read_data(data_dir, batch_size):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle= True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle= True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle= True)
    
    return trainloader, validloader, testloader, train_dataset.class_to_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(image)
    img_tensor.numpy().transpose((1,2,0))
    img_tensor.unsqueeze_(0)

    return img_tensor

def get_cat_to_name(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name
        
def get_idx_to_name(class_to_idx, cat_to_name):
    return {v:cat_to_name[k] for k, v in class_to_idx.items()}