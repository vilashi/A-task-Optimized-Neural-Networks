"""
Implementation of single task Neural Network 
to be optimized for particular task. As mentioned in 
the base paper, it is the first step of the mentioned STAR methods.

The input to the network was a cochleagram of each training exemplar.

Cochleagram:kelletal - 2018

" A cochleagram is a time-frequency decomposition of a sound
that mimics aspects of cochlear processing – it is similar to a spectrogram, 
but with a frequency resolution like that thought to be
present in the cochlea, and with a compressive nonlinearity 
applied to the amplitude in each time-frequency bin. "

-- kelletal - 2018

The file contains sperate neural networks optimized 
and trained to perform particular task only (here one for Music-genre classification
and word classification)self.fc_two(out4)
        
"""

#Importing necessary library functions
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# Defining CNN for music genre classification

from torch.autograd import Variable
import torch.nn.functional as F

class Music_genre_CNN(torch.nn.Module):
    
    def __init__(self, x_len, y_len, kernel, stride_len, pad, output_dim):
        super(Music_genre_CNN, self).__init__()
        self.conv_one = torch.nn.Conv2d(x_len, y_len, kernel_size=kernel, stride=stride_len, padding=pad)
        self.poo_one = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc_one = torch.nn.Linear(y_len * x_len, output_dim)
        
        # here we use output_dim to denote number of classes of generes. 
        self.fc_teo = torch.nn.Linear(x_len, output_dim)
        
    def forward(self, x):
        
        # sequential steps of forward musical input in network
        x = F.relu(self.conv_one(x))
        
        # pooling layer
        x = self.pool(x)
        
        x = F.relu(self.fc_one(x))
        
        x = self.fc_two(x)
        return(x)


class Word_prediction(torch.nn.Module):
    
    # CNN for word classification
    # Reference: https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    def __init__(self):
        super(Word_prediction, self).__init__()
        self.layer_one = torch.nn.Sequential(torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer_two = torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc_one = torch.nn.Linear()
        self.fc_two = torch.nn.Linear()
    
    def forward(self, x):
        out1 = self.layer_one(x)
        out2 = self.layer_two(out1)
        out3 = out2.reshape(out2.size(0), -1)
        out4 = self.fc_one(out3)
        return self.fc_two(out4)
        

#Dataset decorator and dataloader

class Cochelogram_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, dir = 'Datasets/data_all.npz'):
        # Load dataset from Numpy archive
        X = np.load(dir)
        # Load Input data from each classes stored as dictionary
        # Remeber: Order here does not matter as we do different class classification
        self.samples = np.concatenate((X['z'], X['s'], X['f'], X['n'], X['o']))
        # Prepare class labels
        self.labels = np.concatenate((np.repeat(1.0, 100), np.repeat(2.0, 100), np.repeat(3.0, 100), np.repeat(4.0, 100), np.repeat(5.0, 100)))

    def __len__(self):
        return self.samples.shape[0]

    # getter for the dataset
    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
