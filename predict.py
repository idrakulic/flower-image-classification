import torch, torchvision
from net import *
from parse_args import *
from read_data import *

def get_device(gpu):
    return torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    
def main():
    in_arg = get_input_args_predict()
    print_command_line_arguments_predict(in_arg)
    device = get_device(in_arg.gpu)
    print(f"Device is {device}")
    cat_to_name = get_cat_to_name(in_arg.category_names)    
    #print(cat_to_name)
    net = Net()
    model = net.load_model(in_arg.checkpoint, device)

    #load_image
    image = Image.open(in_arg.input).convert('RGB')
    #resize_image
    np_image = process_image(image)
    #predict class by image
    model.eval()
    pblogs = model(np_image)
    probabilities = torch.exp(pblogs)
    
    #get top_k klasses
    top_ps, top_class = probabilities.topk(in_arg.top_k, dim=1)
    probabilities = top_ps[0].detach().numpy()

    idx_to_names = get_idx_to_name(model.class_to_idx, cat_to_name)
    class_names = [idx_to_names.get(key) for key in top_class[0].numpy()]
    print('----------------------------------------------------')
    print(f"Most probable class name is {class_names[0]} with probability of {probabilities[0]}")
    print('----------------------------------------------------')
    print(f'Top {in_arg.top_k} classes and their probabilities')
    for c,p in zip(class_names, probabilities):
        print(f'Class {c:25} with probability {p}')


# Call to main function to run the program
if __name__ == "__main__":
    main()