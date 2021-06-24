import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image,ImageOps
import torchvision
from torch import optim,nn
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tqdm
import os

import warnings
warnings.filterwarnings('ignore')

## For creating dataset from inages
class Xray_dataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        tag = torch.tensor(self.df.iloc[idx, 1]-1)

        if self.transform:
            image = self.transform(image)

        return (image,tag)
    
## Training Model using Validation
def train_model(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # initialize weights to zero
            optimizer.zero_grad()
            
            output = model(data)
            
            # calculate loss
            loss = criterion(output, target)
            
            # back prop
            loss.backward()
            
            # grad
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            if batch_idx % 50 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
        
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['val']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model, save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            
    # return trained model
    return model

## Testing Model
def test_model(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
## Gradcam for resnet  
def grad_cam_resnet(model,img,target_layer):
    feture = []
    def fet(m,i,o):
        feture.append(o)
    grad = []
    def grd(m,i,o):
        grad.append(o[0])
    h = target_layer.register_backward_hook(grd)
    g = target_layer.register_forward_hook(fet)
    y = model(img)
    y[:,y.argmax(dim=1).item()].backward()
    h.remove()
    g.remove()
    
    w = torch.mean(grad[-1],dim = [2,3])
    cam = w[:,:,None,None]*feture[-1]
    cam = torch.mean(cam,dim = 1).squeeze().detach().cpu()
    cam = np.maximum(cam,0)
    cam/= torch.max(cam)
    cam = cam.numpy()
    cam = cv2.resize(cam,(256,256))
    cam = np.uint8(255*cam)
  
    img = img.squeeze().cpu()

    plt.imshow(torch.mean(img, dim = 0),cmap = 'gray')
    plt.imshow(cam,alpha = 0.4) 


## Plot GradCam 
def gradcam(model,image):
    im = image.clone().detach()
    image.requires_grad = True
    image = model.features(image)
    activation = image
    x = []
    def f(g):
        x.append(g) 
    h = image.register_hook(f)

    image = nn.AvgPool2d(7)(image)
    image = image.view(image.size(0), -1)
    image = model.classifier(image)
    target = np.argmax(image.cpu().data.numpy())
    image[:,target].backward()

    h.remove()
    w = torch.mean(x[0],dim = [2,3])
    cam = w[:,:,None,None]*activation
    cam = torch.mean(cam,dim = 1).squeeze().detach().cpu()
    cam = np.maximum(cam,0)
    cam/= torch.max(cam)
    cam = cam.numpy()
    cam = cv2.resize(cam,(256,256))
    cam = np.uint8(255*cam)
  
    im = im.squeeze().cpu()

    plt.imshow(torch.mean(im, dim = 0),cmap = 'gray')
    plt.imshow(cam,alpha = 0.4)
    
## Print Confusion matrix
def conf_met(model,test_loader):
    true = []
    pred = []
    test_loss = 0
    for img,tag in tqdm.tqdm(test_loader):
        img = img.cuda()
        op = model(img)
        #loss = criterion(output, target)
        #test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        pred+= list(op.data.max(1, keepdim=True)[1].cpu().squeeze().data.numpy())
        true += list(tag.data.numpy())

    print(f"\nConfusion Metrix : \n{confusion_matrix(true,pred)}")

###Print classification matrix    
def class_met(model,test_loader):
    true = []
    pred = []
    test_loss = 0
    for img,tag in tqdm.tqdm(test_loader):
        img = img.cuda()
        op = model(img)
        #loss = criterion(output, target)
        #test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        pred+= list(op.data.max(1, keepdim=True)[1].cpu().squeeze().data.numpy())
        true += list(tag.data.numpy())

    print(f"\nClassification Report : \n{classification_report(true,pred)}")
    