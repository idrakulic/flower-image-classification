
import torch, torchvision
from torchvision import transforms, datasets, models
from torch import nn, optim
from net import *
from parse_args import *
from read_data import *
import os
arch_input_size = {'vgg13': 25088,'resnet50':2048, 'densenet121':1024}

def get_device(gpu):
    return torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

def add_new_arch(arch, input_size):
    if arch not in arch_input_size.keys():
        arch_input_size[arch] = input_size
    
def main():
    """
    Creates a model based on parameters and loads it from existing pretrained models, freezes existing weights,
    replaces the classifier with our network and trains changing only weights of classifier. Saves the checkpoint to pth file
    Parameters: 
     
    data_dir- The path to the folder of images that are to be classified 
    --arch - name of the architecture choosen may be one of 'vgg13', 'resnet50' or 'densenet121' (default is vgg13)
        other architectures can be used only if number of input parameters to the classifier is provided in --input_size argument
    --input_size - optional argument to be used if other than previous three architectures are wanted for example: --arch 'vgg16'     --input_size 25088
    --save_dir - path to directory where checkpoints will be saved  (default 'saved_models')
    --learning_rate - learning rate - float value (default 0.003)
    --hidden_units - number of hidden units in classifier - int value (default 128)
    --epochs - number of epochs-int value (default 5)
    --gpu - indicates if gpu should be used, takes no value, just by using --gpu tries to enable it if  cuda is available
    Returns:
    None     
    """
    in_arg = get_input_args_train()
    print_command_line_arguments_train(in_arg)
    device = get_device(in_arg.gpu)
    print(f"Device is {device}")
    if in_arg.input_size != 0:
        add_new_arch(in_arg.arch, in_arg.input_size)
        
    in_arg.data_directory = in_arg.data_directory.rstrip('/')
    in_arg.save_dir = in_arg.save_dir.rstrip('/')

    if not os.path.exists(in_arg.save_dir):
        os.makedirs(in_arg.save_dir)
        print(f"The directory {in_arg.save_dir} is created.")

    print(f'Possible arch to chose from are {arch_input_size}')

    
    trainloader, validloader, testloader, class_to_idx = read_data(in_arg.data_directory, in_arg.batch_size)
    model = Net()
    model.create_model(in_arg.arch, in_arg.hidden_units, in_arg.learning_rate, arch_input_size[in_arg.arch], class_to_idx, device)
    
    model.train_model(in_arg.epochs, trainloader, validloader)
    model.test_model(testloader, 'Test')
    model.save_model(in_arg.save_dir)

# Call to main function to run the program
if __name__ == "__main__":
    main()
