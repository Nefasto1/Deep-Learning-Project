#####################################################
###                                               ###
###  File containing Model functions and classes  ###
###                                               ###
#####################################################
# Importing libraries
import numpy as np
import torch as th

from torch.utils.data import Dataset

from tqdm import trange

# For reproducibility
th.manual_seed(3407)

##################################################################################################################
################################################ MODEL DEFINITION ################################################
##################################################################################################################

## ------------------------------------------------------------------------------------------------------------ ##
## ------------------------------------------------- Dataset -------------------------------------------------- ##
## ------------------------------------------------------------------------------------------------------------ ##

class Custom_Dataset(Dataset):
    """
    Class for the Geometric Figures Dataset

    Attributes
    ----------
    images: th.Tensor
        Tensor containing the images information
        The shape should be (num_images, width, height, num_channels)
    annotations: th.Tensor
        Tensor containing the labels 
    relationships. th.Tensor
        Tensor containing the relationships between the shapes
    """
    def __init__(self, 
                 images: th.Tensor, 
                 annotations: th.Tensor, 
                 relatioships: th.Tensor):
        self.images = images.permute(0, 3, 1, 2)
        self.annotations = annotations
        self.relationships = relatioships

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        annotation = self.annotations[idx]
        relatioships = self.relationships[idx]
        
        return image, annotation, relatioships


## ------------------------------------------------------------------------------------------------------------ ##
## ------------------------------------------- Convolutional Layer Block -------------------------------------- ##
## ------------------------------------------------------------------------------------------------------------ ##

def make_conv_block(in_channel: int, 
                    out_channel: int, 
                    kernel_size: int=3):
    """
    Function to create a convolutional block containing a convolutional layer, a BatchNormalization Layer and a MaxPool layer
    Reduces to half the image size

    Parameters
    ----------
    in_channels: int
        Number of input channel
    out_channels: int
        Number of output channel
    kernel_size: int (Optional)
        Size of the kernel size
    """
    padding = (kernel_size - 1) // 2
    return th.nn.Sequential(
        th.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        th.nn.BatchNorm2d(out_channel),
        th.nn.MaxPool2d(kernel_size=2, stride=2),
        th.nn.LeakyReLU(0.1),        
    )


## ------------------------------------------------------------------------------------------------------------ ##
## ---------------------------------------------- Scene Graph Model ------------------------------------------- ##
## ------------------------------------------------------------------------------------------------------------ ##

class Scene_Graph_Model(th.nn.Module):
    """
    Class for the YOLO-like architecture  
    """
    def __init__(self, 
                 num_boxes: int, 
                 num_classes: int, 
                 num_relation: int):
        super(Scene_Graph_Model, self).__init__()

        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.num_relation = num_relation

        # CNN block to obtain informations from the images
        self.block1 = th.nn.Sequential(
           make_conv_block(3, 16),                    # (3, 128, 128) -> (16, 64, 64)
           make_conv_block(16, 32),                   # (16, 64, 64)  -> (32, 32, 32)
           make_conv_block(32, 64),                   # (32, 32, 32)  -> (64, 16, 16)
           make_conv_block(64, 128),                  # (64, 16, 16)  -> (128, 8, 8)
           make_conv_block(128, 256, kernel_size=1),  # (128, 8, 8)    -> (256, 4, 4)
           make_conv_block(256, 512, kernel_size=1),  # (256, 4, 4)    -> (256, 2, 2)
           make_conv_block(512, 1024, kernel_size=1), # (256, 2, 2)    -> (256, 1, 1)
        )

        self.block_classification = th.nn.Sequential(
            th.nn.Linear(in_features=1024, out_features=512),
            th.nn.ReLU(),
            th.nn.Linear(in_features=512, out_features=256),
            th.nn.ReLU(),
            th.nn.Linear(in_features=256, out_features=128),
            th.nn.ReLU(),
            th.nn.Linear(in_features=128, out_features=num_boxes * (num_classes + 1)),
        )

        self.block_regression = th.nn.Sequential(
            th.nn.Linear(in_features=1024, out_features=512),
            th.nn.ReLU(),
            th.nn.Linear(in_features=512, out_features=256),
            th.nn.ReLU(),
            th.nn.Linear(in_features=256, out_features=128),
            th.nn.ReLU(),
            th.nn.Linear(in_features=128, out_features=num_boxes * 4),
        )
        
        self.block_relation = th.nn.Sequential(
            th.nn.ReLU(),
            th.nn.Linear(in_features=num_boxes * 4, out_features=1024),
            th.nn.ReLU(),
            th.nn.Linear(in_features=1024, out_features=512),
            th.nn.ReLU(),
            th.nn.Linear(in_features=512, out_features=256),
            th.nn.ReLU(),
            th.nn.Linear(in_features=256, out_features=128),
            th.nn.ReLU(),
            th.nn.Linear(in_features=128, out_features=num_relation * num_boxes * num_boxes)
        )

    def forward(self, x):
        # CNN Block
        x = self.block1(x)
        x = x.flatten(1)

        # Information Blocks
        x_classification = self.block_classification(x)
        x_regression = self.block_regression(x)
        x_relation = self.block_relation(x_regression)

        # Reshape of the outputs
        x_classification = x_classification.view(-1, self.num_boxes, (self.num_classes + 1))
        x_regression = x_regression.view(-1, self.num_boxes, 4)
        x_relation = x_relation.view(-1, self.num_relation, self.num_boxes, self.num_boxes)

        # Concatenation of the informations
        return th.cat((x_classification, x_regression), dim=2), x_relation
