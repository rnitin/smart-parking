# Program to train a CNN with PyTorch by transfer learning on ResNet50

# ***** Import libraries *****
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import time
from PIL import Image
import shutil
import os


# ***** Function definitions *****

def image_convert(img):
    '''
    Function to process image for display
    Parameter: img (Torch tensor)
    Returns: image (numpy array)  
    '''
    img = img.clone().cpu().numpy()
    img = img.transpose(1, 2, 0)
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    img = img*std + mean
    return img

def train_model(model, loss_criterion, optimiser, epochs):
    '''
    Function to train a Torch model on training set
    Parameter: model (Torch model - untrained), loss_criterion (Torch class), optimiser (Torch class), epochs (int) 
    Returns: model (Torch model - trained), history (list)
    '''
    history = [] # list to record model training trend

    for epoch in range(epochs): # iterate through epochs
        start_time = time.time() # record computer time at the instant
        print("Epoch: {}/{}".format(epoch+1, epochs))

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        
        # Training loop
        for i, (inputs, labels) in enumerate(train_loader): # iterate through the training batches
            model.train() # set model to training mode
            inputs = inputs.to(device) # send images to device
            labels = labels.to(device) # send labels to device
            optimiser.zero_grad() # remove existing gradients
            outputs = model(inputs) # compute outputs for the training inputs
            loss = loss_criterion(outputs, labels) # compute loss of the obtained outputs       
            loss.backward() # backpropagate the gradients
            optimiser.step() # update parameters
            
            train_loss += loss.item() * inputs.size(0) # compute and add the total loss of the batch       
            ret, predictions = torch.max(outputs.data, 1) # compute accuracy
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor)) # compute mean accuracy
            train_acc += acc.item() * inputs.size(0) # compute and add total accuracy of the batch
            print("Batch number: " + str(i))
            
        # Validation loop
        with torch.no_grad():
            model.eval() # set model to evaluation mode
            # Validation loop
            for j, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device) # send images to device
                labels = labels.to(device) # send labels to device
                outputs = model(inputs) # compute outputs for the present inputs
                loss = loss_criterion(outputs, labels) # compute loss
                valid_loss += loss.item() * inputs.size(0) # compute and add the total loss of the batch
                ret, predictions = torch.max(outputs.data, 1) # compute accuracy
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor)) # compute mean accuracy
                valid_acc += acc.item() * inputs.size(0) # compute and add total accuracy of the batch
                print("Validation Batch number: " + str(j))

        avg_train_loss = train_loss/dataset_sizes[0] # find average training loss
        avg_train_acc = train_acc/dataset_sizes[0] # find average training accuracy
        avg_valid_loss = valid_loss/dataset_sizes[1] # find average validation loss
        avg_valid_acc = valid_acc/dataset_sizes[1] # find average validation accuracy
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])               
        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_time))
        
    return model, history

def test_accuracy(model, loss_criterion):
    '''
    Function to compute the accuracy of trained model on test set
    Parameter: model (Torch model - trained)
    Returns: none
    '''
    test_acc = 0.0
    test_loss = 0.0

    with torch.no_grad(): # turn off gradient tracking
        model.eval() # set model to evaluate mode
        for j, (inputs, labels) in enumerate(test_loader): # iterate through test set batches
            inputs = inputs.to(device) # send images to device
            labels = labels.to(device) # send labels to device
            outputs = model(inputs) # compute outputs for the present inputs
            loss = loss_criterion(outputs, labels) # compute loss
            test_loss += loss.item() * inputs.size(0) # compute and add the total loss of the batch
            ret, predictions = torch.max(outputs.data, 1) # compute accuracy
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor)) # compute mean accuracy
            test_acc += acc.item() * inputs.size(0) # compute and add total accuracy of the batch
            print("Test batch number: " + str(j))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/dataset_sizes[2]
    avg_test_acc = test_acc/dataset_sizes[2]
    print("Test accuracy: " + str(avg_test_acc))
    print("Test loss: " + str(avg_test_loss))


# ***** Loading datasets *****

# Initialise training and test set directories
root_dir = './train' # insert root dir here
result_dir = root_dir + '/results/'
training_set = root_dir + '/dataset/carpark/training_set'
valid_set = root_dir + '/dataset/carpark/valid_set'
test_set = root_dir + '/dataset/carpark/test_set'

# Define transformations for the train, validation and test sets
transform_train = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness = 0.5, contrast = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # image_net values
])
transform_valid = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # image_net values
])
transform_test = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # image_net values
])

# Load the datasets from their folders to the dataloader objects
trainset = datasets.ImageFolder(training_set, transform = transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)

validset = datasets.ImageFolder(valid_set, transform = transform_valid)
valid_loader = torch.utils.data.DataLoader(validset, batch_size = 64, shuffle = True)

testset = datasets.ImageFolder(test_set, transform = transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)

# Obtain the output class names and the dataset sizes
class_names = trainset.classes
dataset_sizes = [len(trainset), len(validset), len(testset)]

# Print sample images from training set
examples = enumerate(train_loader)
batch_id, (images, targets) = next(examples) # read a batch from the training set
fig = plt.figure() # create a matplotlib pyplot figure
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.tight_layout()
    plt.imshow(image_convert(images[i])) # convert tensor to image and plot
    plt.title(class_names[targets[i].item()]) # targets is a tensor storing the corresponding class of images
plt.show()


# ***** Model and parameter definition *****

model = models.resnet50(pretrained=True) # load pretrained ResNet50 Model
for param in model.parameters():
    param.requires_grad = False # freeze model parameters
# summary(model, (3, 224, 224), 32, device='cpu') # Display model summary in tensorflow style

fc_inputs = model.fc.in_features # define final layers for transfer learning 
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2),
    nn.LogSoftmax(dim=1) # To use NLLLoss()
)

# Define Optimiser and Loss Function
loss_func = nn.NLLLoss()
optimiser = optim.Adam(model.parameters())

#Check if CUDA is available, or use CPU as the training device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device) # send model to CUDA device


# ***** Model training ******

model_no = '1' # change on each iteration
num_epochs = 10 # 10 epochs
trained_model, history = train_model(model, loss_func, optimiser, num_epochs) # train the model
torch.save(model, result_dir + 'carpark_' + model_no + '.pt') # save trained model

test_accuracy(model, loss_func) # test model accuracy on the test set


# ***** Plotting loss and accuracy curves *****
history = np.array(history)

# Loss curve
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig(result_dir + 'loss_curve_' + str(model_no) + '.png') # save loss curve in result directory
plt.show()

# Accuracy curve
plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(result_dir + 'accuracy_curve_' + str(model_no) + '.png') # save accuracy curve in result directory
plt.show()