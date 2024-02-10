import argparse
def get_input_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, default='flowers', 
                        help='path to images folder')
    parser.add_argument('--arch', type=str, default='vgg13', 
                        help='CNN Model')
    parser.add_argument('--save_dir', type=str, default='saved_models', 
                        help='Directory of saved models')
    parser.add_argument('--learning_rate', type=float, default=0.003, 
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=128,
                        help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of epochs')		
    parser.add_argument('--gpu', type=bool,  nargs="?", default=False, const=True,
                        help='Usage of gpu')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--input_size', type=int, default=0, 
                        help='Input size')
    return parser.parse_args()

def print_command_line_arguments_train(in_arg):
    if in_arg is None:
        print("input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:"
              "\n       data_directory =", in_arg.data_directory, 
              "\n       arch =", in_arg.arch, 
              "\n       save_dir =", in_arg.save_dir,
              "\n       learning_rate =", in_arg.learning_rate, 
              "\n       hidden_units =", in_arg.hidden_units,
              "\n       epochs =", in_arg.epochs, 
              "\n       gpu =", in_arg.gpu,
              "\n       batch_size =", in_arg.batch_size,
              "\n       input_size =", in_arg.input_size)	
        
def get_input_args_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='assets/Flowers.png', 
                        help='Path to image')
    parser.add_argument('checkpoint', type=str, default='', 
                        help='Model checkpoint')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Json file with names')
    parser.add_argument('--gpu', type=bool,  nargs="?", default=False, const=True, help='Usage of gpu')
    return parser.parse_args()

def print_command_line_arguments_predict(in_arg):
    if in_arg is None:
        print("input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:"
              "\n    input =", in_arg.input, 
              "\n    checkpoint =", in_arg.checkpoint,
              "\n    top_k =", in_arg.top_k,
              "\n    category_names =", in_arg.category_names, 
              "\n    gpu =", in_arg.gpu)