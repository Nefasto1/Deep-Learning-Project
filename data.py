######################################################################
###                                                                ###
###  File containing functions for data generation and management  ###
###                                                                ###
######################################################################

# Importing libraries

import os
import platform
import subprocess

import numpy as np
import torch as th

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import GeometricFiguresDataset
from torch.utils.data import DataLoader

from tqdm import trange

# For reproducibility
th.manual_seed(3407)



def create_dataset(num_images: int, image_size: int, image_dir: str, max_objects: int, shape_size_min: int, shape_size_max: int):
    """
    Function to generate some images with geometric shapes. These will be generated avoiding the overlaps and will generate the relationship matrix.
    The relationships are: "Left", "Right", "Above", "Below", "In Front of", "Behind"

    Parameters
    ----------
    num_images: int
        Number of images to generate
    image_size: int
        Size of the squared images
    image_dir: str
        Directory where to save the images (if not exist will be created)
    max_objects: int
        Number of the shapes inserted in the images
    shape_size_min: int
        Minimum size of the shapes
    shape_size_max: int
        Maximum size of the shapes

    Returns
    -------
    data: torch.Tensor
        The label dataset of the images informations
    relationships: torch.Tensor
        Tensor containing the relatioships between the shapes in the images
    """
    ###########################
    ## REMOVE PREVIOUS FILES ##
    ###########################
    def remove_previous_files():
        """
        Remove all the previous informations before to start checking the OS in which we are operating
        """
        if image_dir in os.listdir():
            print(f"The directory {image_dir} already exists. Do you want to delete it? [y/n]")
            if input() in ['y', 'Y']:
                if platform.system() == 'Linux':
                    subprocess.run(['rm', image_dir, '-rf'])
                    # Create directories
                    os.makedirs(image_dir, exist_ok=True)
                    os.makedirs('data', exist_ok=True)
                elif platform.system() == 'Windows':
                    subprocess.run(['rmdir /Q /S', image_dir])
                    # Create directories
                    os.makedirs(image_dir, exist_ok=True)
                    os.makedirs('data', exist_ok=True)

    ####################
    ## CHECK OVERLAPS ##
    ####################
    def is_overlapping(xs: int, ys: int):
        """
        Check if the new shape overlaps with any previous shapes.
        
        Parameters
        ----------
        xs: int
            x-axis coordinate of the center
        ys: int
            y-axis coordinate of the center
        
        Returns
        -------
            Returns True if the two shapes overlaps, False otherwise
        """
        
        for x, y in zip(xs, ys):
            # Check the Euclidean distance between the centers
            if (np.sqrt((x-xs)**2 + (y-ys)**2) < limit).sum() >= 2:
                return True
    
        return False

    ########################
    ## CHECK RELATIOSHIPS ##
    ########################
    def adjacency_matrix(xs: int, ys: int, radius: np.array, widths: np.array, heights: np.array):
        """
        Create the relatioship matrix.

        Parameters
        ----------
        xs: int
            x-axis coordinate of the center
        ys: int
            y-axis coordinate of the center
        radius: np.array
            Numpy array containing the radius of the circles
        widths: np.array
            Numpy array containing the widths of the circles
        heights: np.array
            Numpy array containing the heights of the circles

        Returns
        -------
        relationships: th.Tensor
            Tensor containing of size relationships * num_objects * num_objects which contains
            one if the relationship between the two objects exists
            zero otherwise        
        """        
        ####################
        ## INITIALIZATION ##
        ####################
        # Initialize the Tensors
        adj_matrix = th.zeros((max_objects, max_objects))
        relations = th.zeros((6, max_objects, max_objects))

        # Calculate the Areas of the bounding boxes
        new_widths = np.concatenate((radius*2, widths))
        new_heights = np.concatenate((radius*2, heights))
        areas = new_widths*new_heights

        ###############
        ## ADJACENCY ##
        ###############
        # Check if two elements are adjacent
        for i, (x, y) in enumerate(zip(xs, ys)):
            tmp = list( (np.sqrt((x-xs)**2 + (y-ys)**2) > rel_limit).astype(np.int32) )
            while len(tmp) < max_objects:
                tmp += [0]
            adj_matrix[i] = th.tensor( tmp )

        # Set to zero the diagonal
        adj_matrix[range(max_objects), range(max_objects)] = 0

        ###################
        ## RELATIONSHIPS ##
        ###################
        for i, (x, y, area) in enumerate(zip(xs, ys, areas)):
            # Check the relative positions and sizes
            x_sign = list( xs - x )
            y_sign = list( ys - y )
            bigger = list( areas / area )

            # Pads the None objects to zero (one for the ratio)
            while len(x_sign) < max_objects:
                x_sign += [0]
            while len(y_sign) < max_objects:
                y_sign += [0]
            while len(bigger) < max_objects:
                bigger += [1]

            # Convert the lists into tensors
            x_sign = th.tensor( x_sign )
            y_sign = th.tensor( y_sign )
            bigger = th.tensor( bigger )

            # Check the distances
            x_axis = th.abs(x_sign) > rel_limit
            y_axis = th.abs(y_sign) > rel_limit

            # Set the relatioships with the conditions
            relations[0, i] = x_axis * (x_sign < 0) # Left of
            relations[1, i] = x_axis * (x_sign > 0) # Right of
            relations[2, i] = y_axis * (y_sign < 0) # Above of
            relations[3, i] = y_axis * (y_sign > 0) # Below of
            relations[4, i] = bigger < 0.8          # Behind of
            relations[5, i] = bigger > 1.2          # In Front of

        return adj_matrix*relations

    def generate_image():
        """
        Function which generate a singular image        
        """
        #####################
        ## INITIALIZZATION ##
        #####################    
        # Hard-Coded colors, maximum 5 objects at the moment
        colors = ["#ff0000", "#00ff00", "#0000ff", "#000000", "#ff00ff"]

        # Create annotation tensor
        data = th.empty((num_images, max_objects, 8))
        relationships = th.empty((num_images, 6, max_objects, max_objects))
        
        for i in trange(num_images):         
            # Create figure
            plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
            ax = plt.gca()
            ax.set_xlim(0, image_size)
            ax.set_ylim(0, image_size)
            ax.axis('off')
    
            #########################
            ## GENERATE IMAGE INFO ##
            #########################
            # Generate random number of objects and their shape types
            num_objects = np.random.randint(1, max_objects+1)
            shape_type = np.random.randint(0, 3, num_objects)
            
            # Compute how many objects of each type
            n_rect = (shape_type == 0).sum()
            n_circle = (shape_type == 1).sum()
            n_tria = (shape_type == 2).sum()
            
            # Generate random centers until there are not any overlaps
            x_center = np.random.randint(limit//2, image_size - limit//2, num_objects)
            y_center = np.random.randint(limit//2, image_size - limit//2, num_objects)
            while is_overlapping(x_center, y_center):
                x_center = np.random.randint(limit//2, image_size - limit//2, num_objects)
                y_center = np.random.randint(limit//2, image_size - limit//2, num_objects)
    
            # Generate random sizes for each shape (circles need only the radius
            width  = np.random.randint(shape_size_min, shape_size_max, n_rect+n_tria)
            height = np.random.randint(shape_size_min, shape_size_max, n_rect+n_tria)
            radius = np.random.randint(shape_size_min/2, shape_size_max/2, n_circle)
    
            # Generate the relationships
            relationships[i] = adjacency_matrix(x_center, y_center, radius, width, height)
    
            ##########################
            ## ADD PATCHES TO IMAGE ##
            ##########################
            # Add the circles
            for row in range(n_circle): 
                x = x_center[row]
                y = y_center[row]
                r = radius[row]
    
                ax.add_patch(patches.Circle((x, y), r, facecolor=colors[row], linewidth=3))
                data[i, row] = th.tensor( [0, 0, 1, 0, x, y, r*2 + np.random.randint(5, 10), r*2 + np.random.randint(5, 10)] )
    
            # Add the rectangles
            for row in range(n_rect): 
                w = width[row]
                h = height[row]
                x = x_center[row+n_circle]
                y = y_center[row+n_circle]
    
                ax.add_patch(patches.Rectangle((x - (w/2), y - (h/2)), w, h, facecolor=colors[row+n_circle], linewidth=3))
                data[i, row+n_circle] = th.tensor( [0, 1, 0, 0, x, y, w + np.random.randint(5, 10), h + np.random.randint(5, 10)] )
    
            # Add the Triangles
            for row in range(n_rect, n_rect+n_tria): 
                w = width[row]
                h = height[row]
                x1, y1 = x_center[row+n_circle] - (w/2), y_center[row+n_circle] - (h/2)
                x2, y2 = x_center[row+n_circle] + (w/2), y_center[row+n_circle] - (h/2)
                x3, y3 = x_center[row+n_circle]        , y_center[row+n_circle] + (h/2)
    
                ax.add_patch(patches.Polygon([[x1, y1], [x2, y2], [x3, y3]], closed=True, facecolor=colors[row+n_circle], linewidth=3))
                data[i, row+n_circle] = th.tensor( [0, 0, 0, 1, x_center[row+n_circle], y_center[row+n_circle], w + np.random.randint(5, 10), h + np.random.randint(5, 10)] )
    
            # Add the None Objects for padding
            for row in range(n_rect+n_circle+n_tria, max_objects):
                data[i, row] = th.tensor( [1, 0, 0, 0, 0, 0, 0, 0] )

            #################
            ## SAVE IMAGES ##
            #################
            # Save image
            string = i + '.jpg' if type(i) == str else f'{i:04d}.jpg'
            image_path = os.path.join(image_dir, string)
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig(image_path, dpi=100)
            plt.close()

        return data, relationships

    
    ###################
    ## MAIN FUNCTION ##
    ###################
    # Define the limits for overlapping and the relationships distance
    limit:int = shape_size_max * 1.2
    rel_limit:int = shape_size_max / 2
    
    remove_previous_files()
    # Create the directory if not exists
    os.makedirs(image_dir, exist_ok=True)
    
    return generate_image()




def data_visualization(train_image_dir: str = "./train_images", train_data_dir: str = "./data/train_data.tensor"):
    """
    Pick 4 random images to display and draw bounding boxes

    Parameters
    ----------
    train_image_dir: str
        Directory where the train images are stored
    train_data_dir: str
        Directory where the train data are stored
    """
    
    train_data = th.load(train_data_dir)
    for i in np.random.choice(range(train_data.shape[0]), 4, replace=False):
        # Load image
        image_path = os.path.join(train_image_dir, f'{i:04d}.jpg')
        image = plt.imread(image_path)
    
        plt.figure(figsize=(5, 5))
    
        # Display image
        plt.imshow(image[::-1], origin='lower')
        # Get the current reference
        ax = plt.gca()
        plt.axis('off')
        
        # Draw centers
        for shape in train_data[i]:
            # If is a shape
            if shape[0] != 1:
                x, y = shape[4], shape[5]
                w = shape[6]
                h = shape[7]
    
                # Add text
                text = 'Rectangle'
                if shape[2] == 1:
                    text = 'Circle'
                elif shape[3] == 1:
                    text = 'Triangle'
                    
                ax.scatter(x, y, color='purple', s=10)
                ax.add_patch(patches.Rectangle((x - (w/2), y - (h/2)), w, h, facecolor="None", edgecolor="purple", linewidth=3))
                ax.text(x - (w/2), y - (h/2) - 7, text, color='purple', fontsize=12)
    
        # Show images
        plt.show()

def make_DataLoader(train_image_dir: str, test_image_dir: str, image_size: int, data_dir: str, train_annotations: str, test_annotations: str):
    """
    Function which convert the images dataset into a Dataloader and save it in storage

    Parameters
    ----------
    train_image_dir: str
        Directory where to load the train images
    test_image_dir: str
        Directory where to load the test images
    image_size: int
        Size of the images
    data_dir: str
        Directory where to load/save the data
    train_annotations: str
        Directory where to load the train annotations
    test_annotations: str
        Directory where to load the test annotations
    """

    def load_images(img_dir, img_size):
        """
        Function to load the images from a directory

        Parameters:
        -----------
        img_dir: str
            Directory where the images are
        img_size: int
            Size of the images

        Returns
        -------
        images: th.Tensor[float]
            Tensor of size n_images * image_size * image_size * n_channels containing the values of the images
        """

        # Retrieve the data into the directory (ordered by name)
        file_list = sorted(os.listdir(img_dir))
        N = len(file_list)

        # Initialize the final Tensor
        images = np.empty((N, img_size, img_size, 3))

        # Load the images singularly
        for i in trange(N, desc="Loading the images in " + img_dir + "..."):
            images[i] = plt.imread(img_dir + "/" + file_list[i])#.mean(axis=2, keepdims=True) # From 3 channels to 1 channel (grayscale)

        # convert the images from numpy array to torch tensor
        images = th.from_numpy(images)
        return images
        

    ###############
    ## LOAD DATA ##
    ###############
    # Load the train and test images
    X_train = load_images(train_image_dir, image_size)
    X_test = load_images(test_image_dir, image_size)
    
    # Load the train and test data
    y_train = th.load(data_dir + "/" + train_annotations)
    y_test = th.load(data_dir + "/" + test_annotations)

    # Load the train and test relatioships
    test_relationships = th.load(data_dir + "/" + str(test_annotations).replace("data", "relationships"))
    train_relationships = th.load(data_dir + "/" + str(train_annotations).replace("data", "relationships"))

    ####################
    ## NORMALIZE DATA ##
    ####################
    # Evaluate the mean and the variances of the images on the train set and normalize the images
    mu, std = X_train.mean(axis=(0, 1, 2)), X_train.std(axis=(0, 1, 2))
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std

    # Save the normalization parameters
    th.save(mu, data_dir + "/mu")
    th.save(std, data_dir + "/std")

    ######################
    ## GENERATE DATASET ##
    ######################
    # Define the torch Dataset
    train_dataset = GeometricFiguresDataset(X_train, y_train, train_relationships)
    test_dataset = GeometricFiguresDataset(X_test, y_test, test_relationships)

    # Free the memory
    del X_train
    del y_train
    del X_test
    del y_test

    os.makedirs('datasets', exist_ok=True)

    # Save the dataset into the storage
    th.save(train_dataset, "./datasets/train_dataset.tensor")
    th.save(test_dataset, "./datasets/test_dataset.tensor")

    
    #########################
    ## GENERATE DATALOADER ##
    #########################
    # Define the Batch size
    BATCH_SIZE = 256

    # Define the torch DataLoader
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    # Free the memory
    del train_dataset
    del test_dataset
    
    # Create directory to save dataloaders
    os.makedirs('dataloaders', exist_ok=True)
    
    # Save dataloaders
    th.save(train_loader, './dataloaders/train_loader.tensor')
    th.save(test_loader, './dataloaders/test_loader.tensor')