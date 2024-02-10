import torch, torchvision
from torchvision import models
from torch import nn, optim

class Net:
    def __init__(self):
        pass
        
    def get_classifier(self):
        classifier_name = [n for n,_ in self.model.named_children()][-1]
        return getattr(self.model, classifier_name)
    
    def create_model(self, arch, hidden_units, learning_rate, input_size, class_to_idx, device):
        self.arch = arch
        self.hidden_units = hidden_units
        self.learning_rate =learning_rate
        self.input_size = input_size
        self.device = device
        
        new_model = getattr(models, self.arch)
        model = new_model(pretrained = True)
        #gets the name of the classifier layer of the nn, some are 'fc' some 'classifier'
        classifier_name = [n for n,_ in model.named_children()][-1]

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(nn.Linear(self.input_size, self.hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(self.hidden_units,102),
                                       nn.LogSoftmax(dim = 1))
            #sets attribute of the model to classifier (model.fc or model.classifier)
        setattr(model, classifier_name, classifier)
        self.model = model
        self.criterion = nn.NLLLoss()
        self.model.class_to_idx = class_to_idx
        classifier = self.get_classifier()
        self.optimizer = optim.Adam(classifier.parameters(), lr = self.learning_rate)
    
    def test_model(self, loader, test_kind):
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            self.model.eval()
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output=self.model(images)
                loss = self.criterion(output,labels)
                test_loss+=loss.item()

                probabilities = torch.exp(output)
                top_ps, top_class = probabilities.topk(1,dim=1)

                is_equal = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(is_equal.type(torch.FloatTensor)).item()
            self.model.train()
            print(f'{test_kind} Loss {test_loss/len(loader)}, {test_kind} Accuracy {test_accuracy/len(loader)}')
            return (test_loss, test_accuracy)
        
    def train_model(self, epochs, trainloader, validloader):
        #training the model
        self.epochs = epochs
        self.model.to(self.device)

        all_train_losses=[]
        all_val_losses=[]
        all_train_accuracy=[]
        all_val_accuracy=[]

        for e in range(self.epochs):
            train_loss = 0
            train_accuracy = 0

            for images, labels in trainloader:
                self.optimizer.zero_grad() #reset optimizer
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                train_loss+= loss.item()#loss in one epoch cumulative
            else:
                #validation
                val_loss, val_accuracy = self.test_model(validloader, 'Validation')
                self.model.train()

                avg_train_loss = train_loss/len(trainloader)
                avg_val_loss = val_loss/len(validloader)
                avg_train_accuracy = train_accuracy/len(trainloader)
                avg_val_accuracy = val_accuracy/len(validloader)

                all_train_losses.append(avg_train_loss)
                all_val_losses.append(avg_val_loss)
                all_train_accuracy.append(avg_train_accuracy)
                all_val_accuracy.append(avg_val_accuracy)

                print(f'Epoch {e+1}/{self.epochs}')
                print(f'Train loss {avg_train_loss}')
                print(f'Validation loss {avg_val_loss}')
                print(f'Validation Accuracy {avg_val_accuracy}')
                print('------------------------------------------')

    # TODO: Save the checkpoint
    def save_model(self, path_to_file):
        classifier = self.get_classifier()
        checkpoint = {
            'epochs':self.epochs,
            'arch':self.arch,
            'model_state_dict': classifier.state_dict(),
            'hidden_units':self.hidden_units,
            'learning_rate':self.learning_rate,
            #'batch_size':self.batch_size,
            'input_size':self.input_size,
            'class_to_idx':self.model.class_to_idx,
            'optimizer_state_dict':self.optimizer.state_dict()
        }
        file_name = "/model_arch{}_hidden{}_epochs{}.pth".format(self.arch, self.hidden_units,self.epochs)
        full_file_name = path_to_file + file_name
        torch.save(checkpoint, full_file_name)
    
    # TODO: Write a function that loads a checkpoint and rebuilds the model
    def load_model(self, path_to_file, device):
        checkpoint = torch.load(path_to_file)

        self.epochs = checkpoint['epochs']
        self.arch = checkpoint['arch']
        self.hidden_units = checkpoint['hidden_units']
        self.learning_rate = checkpoint['learning_rate']
        class_to_idx = checkpoint['class_to_idx']
        self.input_size = checkpoint['input_size']
        self.create_model(self.arch, self.hidden_units, self.learning_rate, self.input_size, class_to_idx, device)
        
        classifier = self.get_classifier()
        classifier.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return self.model
    
    
    