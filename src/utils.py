##########################################
###                                    ###
###  File containing utility functions ###
###                                    ###
##########################################

# Importing libraries
import numpy as np
import torch as th

from torchmetrics.classification import BinaryF1Score

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import networkx as nx

from prettytable import PrettyTable

from src.data import create_dataset

from tqdm import trange

import os

# For reproducibility
th.manual_seed(3407)

## ------------------------------------------------------------------------------------------------------------ ##
## -------------------------------------------- Count Parameters ---------------------------------------------- ##
## ------------------------------------------------------------------------------------------------------------ ##

def count_parameters(model: th.nn.Module):
    """
    Function to print the number of parameters for each layer of the input model

    Parameters
    ----------
    model: th.nn.Module
        Model to which print the number of parameters
    """

    # Define the table's columns 
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    # For each parameter with learneble parameters add a row to the table
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params

    # Print the table and the total of parameters
    print(table)
    print(f"Total Trainable Params: {total_params}")



## ------------------------------------------------------------------------------------------------------------ ##
## ------------------------------------------------ Plot losses ----------------------------------------------- ##
## ------------------------------------------------------------------------------------------------------------ ##

def plot_loss(train_loss: list, 
              test_loss: list, 
              title: str):
    """
    Function to plot the losses

    Parameters
    ----------
    train_loss: list
        list containing the series of train losses
    test_loss: list
        list containing the series of test losses
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    plt.plot(range(len(train_loss)), train_loss, color="#4e518b", label="Train", linewidth=3)
    plt.plot(range(len(test_loss)), test_loss, color="#9a3001", label="Validation", linewidth=3)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title, fontsize=40, fontdict={"family": "fantasy"})
    plt.rcParams.update({'font.size': 25})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add text near line ends with final value
    # If y values for the two texts are too close, place them one above the other
    if abs(train_loss[-1] - test_loss[-1]) > 50:
        plt.text(len(train_loss), train_loss[-1], f"{train_loss[-1]:.2f}", fontsize=12, color="#4e518b")
        plt.text(len(test_loss), test_loss[-1], f"{test_loss[-1]:.2f}", fontsize=12, color="#9a3001")
    else:
        plt.text(len(train_loss), train_loss[-1], f"{train_loss[-1]:.2f}", fontsize=12, color="#4e518b")
        plt.text(len(test_loss), test_loss[-1] + 50 - abs(train_loss[-1] - test_loss[-1]), f"{test_loss[-1]:.2f}", fontsize=12, color="#9a3001")
        
    plt.show()



## ------------------------------------------------------------------------------------------------------------ ##
## --------------------------------------------- Print Relatioships ------------------------------------------- ##
## ------------------------------------------------------------------------------------------------------------ ##

def print_relationships(relatioships: th.Tensor, 
                        pred_classes: th.Tensor):
    """
    Function to print the relatioships between shapes

    Parameters
    ----------
    pred_positions: th.Tensor
        Tensor containing the relatioships
    pred_classes: th.Tensor
        Tensor containing the classes
    """

    # Initialize the classes
    colors = np.array(["Red", "Green", "Blue", "Black", "Purple"])
    classes = np.array(["None", "Rectangle", "Circle", "Triangle"])
    positions = np.array(["Right", "Left", "Above", "Below", "Front", "Behind"])
    
    classes_names = classes[pred_classes[:, :4].argmax(1)]

    # Print the relationship if exists
    for i, pred_position in enumerate(positions):
        for j, pred_class in enumerate(classes_names):
            for k, target_class in enumerate(classes_names):
                if relatioships[i, j, k]:
                    print(colors[j], pred_class, "is", pred_position, "of", colors[k], target_class)
                    


##################################################################################################################
################################################# MODEL UTILITY ##################################################
##################################################################################################################


## ------------------------------------------------------------------------------------------------------------ ##
## ----------------------------------------------- Train Model ------------------------------------------------ ##
## ------------------------------------------------------------------------------------------------------------ ##

def train_model(model: th.nn.Module, 
                optimizer: th.optim.Optimizer, 
                criterion, 
                train_loader: th.utils.data.DataLoader, 
                test_loader: th.utils.data.DataLoader, 
                epochs:int=1000):
    """
    Function to train the specified model. (Specific for our models)

    Parameters
    ----------
    model: th.nn.Module
        Initialized Model to train
    optimizer: th.optim.Optimizer
        Initialized optimizer to use
    criterion: function
        Function to apply as a loss 
        (must take in input y_hat, y, relationships_hat, relationships)
    train_loader: th.data.DataLoader
        Dataloader containing the train data
    test_loader: th.data.DataLoader
        Dataloader containing the test data
    epochs: int (Optional)
        Number of train epochs

    Returns
    -------
    model: th.nn.Module
        Trained Model
    train_losses: th.Tensor[float]
        Tensor containing the train losses
    test_losses: th.Tensor[float]
        Tensor containing the test losses
    """

    ####################
    ## INITIALIZATION ##
    ####################
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Initialize the final lists
    train_losses          = []
    test_losses           = []
    train_accuracies      = []
    test_accuracies       = []
    train_relationship_f1 = []
    test_relationship_f1  = []
    num_classes = model.num_classes

    # Define the f1 score functional
    f1 = BinaryF1Score().to(device)
    bar = trange(epochs, desc="Loss ?/?, Acc ?/?")
    for i in bar:
        epoch_loss  = 0
        accuracy    = 0
        relation_f1 = 0
        model.train()

        
        ################
        ## TRAIN LOOP ##
        ################
        for image, label, relation in train_loader:
            # Retrieve the train data
            image, label, relation = image.float().to(device), label.to(device), relation.to(device)

            # Run the model on the inputs
            optimizer.zero_grad()
            out, out_relation = model(image)

            # Evaluate the loss
            loss        = criterion(out, label, out_relation, relation)
            epoch_loss += loss.item()

            # Evaluate the different accuracies
            predicted_classes = th.softmax(out[:, :, :num_classes], dim=2).argmax(2)
            true_classes      = label[:, :, :num_classes].argmax(2)
            accuracy         += predicted_classes.eq(true_classes).float().mean().item()        

            relation_f1      += f1((th.sigmoid(out_relation) > 0.5 ).int(), relation.to(device))

            # Update the parameters
            loss.backward()
            optimizer.step()

        # Append the informations
        train_losses.append(epoch_loss/len(train_loader))
        train_accuracies.append(accuracy/len(train_loader))
        train_relationship_f1.append(relation_f1/len(train_loader))
    
        model.eval()
    
        epoch_loss  = 0
        accuracy    = 0
        relation_f1 = 0

        
        #####################
        ## EVALUATION LOOP ##
        #####################
        with th.no_grad():
            for image, label, relation in test_loader:
                # Retrive the test data
                image, label, relation = image.float().to(device), label.to(device), relation.to(device)
    
                # Run the model on the inputs
                out, out_relation = model(image)
                
                # Evaluate the loss
                loss        = criterion(out, label, out_relation, relation)
                epoch_loss += loss.item()
    
                # Evaluate the different accuracies
                predicted_classes = th.softmax(out[:, :, :num_classes], dim=2).argmax(2)
                true_classes = label[:, :, :num_classes].argmax(2)
                accuracy    += predicted_classes.eq(true_classes).float().mean().item()
    
                relation_f1 += f1((th.sigmoid(out_relation) > 0.5 ).int(), relation.to(device))
            
            # Append the informations
            test_losses.append(epoch_loss/len(test_loader))
            test_accuracies.append(accuracy/len(test_loader))
            test_relationship_f1.append(relation_f1/len(test_loader))
                
        # Update the loading bar with the new informations
        bar.set_description(f"Loss {train_losses[-1]:.4f}/{test_losses[-1]:.4f}, Acc {train_accuracies[-1]:.2f}/{test_accuracies[-1]:.2f}, Relation: {train_relationship_f1[-1]:.2f}/{test_relationship_f1[-1]:.2f}") 

    return model, train_losses, test_losses


## ------------------------------------------------------------------------------------------------------------ ##
## ------------------------------------------------ Test Model ------------------------------------------------ ##
## ------------------------------------------------------------------------------------------------------------ ##

def test_model(model: th.nn.Module, 
               device: th.device, 
               data_dir: str, 
               SHAPE_SIZE_MIN: int=10, 
               SHAPE_SIZE_MAX: int=35, 
               geometric: bool = True, 
               rotate: bool = False, 
               origami: bool = False):
    """
    Function to test the model, generate a new image and test the model over that

    Parameters
    ----------
    model: th.nn.Module
        model to test
    device: th.device
        device to use
    data_dir: str
        directory containing the data
    SHAPE_SIZE_MIN: int (Optional)
        Minimum size of the image
    SHAPE_SIZE_MAX: int (Optional)
        Maximum size of the image
    geometric: bool (Optional)
        If True generate images with geometric shapes, if False it uses real image shapes
    rotate: bool (Optional)
        If True generate images with random rotation
        Affect only the Real images
    origami: bool (Optional)
        If True generate images with origami shapes
        Affect only the Real images
    """
    
    ######################
    ## HELPER FUNCTIONS ##
    ######################
    ## ------------------------------------------------------------------------------------------------------------ ##
    ## ----------------------------------------- PREDICTION VISUALIZATION ----------------------------------------- ##
    ## ------------------------------------------------------------------------------------------------------------ ##
    def plot_output(test_image, test_image_data, prediction):
        """
        Function to plot the output of the model

        Parameters
        ----------
        test_image: th.Tensor
            Tensor containing the image
        test_image_data: th.Tensor
            Tensor containing the image
        prediction: th.Tensor
            Tensor containing the model prediction
        """
        f1_function = BinaryF1Score().to(device)
        f1_score = f1_function(th.sigmoid(out_relation) > 0.5, relation.to(device)).item()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        
        # Plot the image and the predicted bounding boxes
        plt.imshow(test_image[::-1], origin='lower')
        plt.axis('off')

        # Create the text mapping for the classes
        if geometric:
            text = ["None", "Rectangle", "Circle", "Triangle"]
        else:
            if origami:
                text = ["None", "Dog", "Fish", "Cat", "Crab"]
            else:
                text = ["None", "Dog", "Fish", "Rathalos", "Bucket"]

        # Plot the bounding boxes, centers and classes
        for i, objects in enumerate(prediction):
            pred_class = np.argmax(objects[:num_classes])
            position   = objects[num_classes:]
            
            if pred_class != 0:
                x, y = position[0], position[1]    
                w = position[2]  
                h = position[3]  
        
                color = "green" if pred_class == test_image_data[0, i, :num_classes].argmax() else "purple"
                ax.scatter(x, y, color=color, s=10) # Centers
                ax.add_patch(patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, facecolor="None", edgecolor=color, linewidth=3)) # Bounding boxes
                ax.text(x - 10, y - 7, text[pred_class], color=color, fontsize=12) # Classes

        ax.set_title(f"Relation F1 Score: {f1_score:.2f}")

        plt.show()

    ## ------------------------------------------------------------------------------------------------------------ ##
    ## ---------------------------------------- RELATIONSHIPS VISUALIZATION --------------------------------------- ##
    ## ------------------------------------------------------------------------------------------------------------ ##
    def plot_relation(out: th.Tensor, 
                      out_relation: th.Tensor):
        """
        Function to plot the relationships graph

        Parameters
        ----------
        out
            Tensor containing the output of the model
        out_relation: th.Tensor
            Tensor containing the output relationships of the model
        """
        # Create figure
        plt.figure(figsize=(15, 15))

        # Map each object in out to its corresponding class and color as string
        if geometric:
            text = ["None", "Rectangle", "Circle", "Triangle"]
            objects = [f"{color} {shape}" 
                       for color, shape 
                       in zip(["Red", "Green", "Blue", "Black", "Purple"], # colors
                              [text[x] for x in out[:, :4].argmax(1) if x] # objects
                )]
        else:
            if origami:
                text = ["None", "Dog", "Fish", "Cat", "Crab"]
            else:
                text = ["None", "Dog", "Fish", "Rathalos", "Bucket"]
            objects = [text[x] for x in test_image_data[:, :5].argmax(1) if x]

        # Process relation data to generate edges
        edges = [[objects[i], objects[j], "is at the right of"] for i, j in [tuple(x) for x in (th.sigmoid(out_relation[:, 0][0]) > 0.5).int().nonzero().tolist()]] + \
                [[objects[i], objects[j], "is at the left of"] for i, j in [tuple(x) for x in (th.sigmoid(out_relation[:, 1][0]) > 0.5).int().nonzero().tolist()]]  + \
                [[objects[i], objects[j], "is above of"] for i, j in [tuple(x) for x in (th.sigmoid(out_relation[:, 2][0]) > 0.5).int().nonzero().tolist()]]        + \
                [[objects[i], objects[j], "is below of"] for i, j in [tuple(x) for x in (th.sigmoid(out_relation[:, 3][0]) > 0.5).int().nonzero().tolist()]]        + \
                [[objects[i], objects[j], "is in front of"] for i, j in [tuple(x) for x in (th.sigmoid(out_relation[:, 4][0]) > 0.5).int().nonzero().tolist()]]     + \
                [[objects[i], objects[j], "is behind of"] for i, j in [tuple(x) for x in (th.sigmoid(out_relation[:, 5][0]) > 0.5).int().nonzero().tolist()]]
            
        # Remove one of the two edges if they are the same relation but in opposite directions
        for edge1 in edges:
            for edge2 in edges:
                if edge1[0] == edge2[1] and edge1[1] == edge2[0] and edge1[2] in ["is at the right of", "is at the left of"] and edge2[2] in ["is at the right of", "is at the left of"]:
                    choice = np.random.choice([0, 1])
                    if choice == 0:
                        edges.remove(edge1)
                        break
                    else:
                        edges.remove(edge2)
                elif edge1[0] == edge2[1] and edge1[1] == edge2[0] and edge1[2] in ["is above of", "is below of"] and edge2[2] in ["is above of", "is below of"]:
                    choice = np.random.choice([0, 1])
                    if choice == 0:
                        edges.remove(edge1)
                        break
                    else:
                        edges.remove(edge2)
                elif edge1[0] == edge2[1] and edge1[1] == edge2[0] and edge1[2] in ["is in front of", "is behind of"] and edge2[2] in ["is in front of", "is behind of"]:
                    choice = np.random.choice([0, 1])
                    if choice == 0:
                        edges.remove(edge1)
                        break
                    else:
                        edges.remove(edge2)
        
        # Create the graph
        G = nx.MultiDiGraph()
        
        # Add edges to the graph
        for edge in edges:
            G.add_edge(edge[0], edge[1], label=edge[2])
        
        # Identify pairs of nodes with multiple edges with different directions
        multiple_edges = [(u, v) for u, v, weight in G.edges(keys=True) if (v, u) in G.edges(keys=True)]
        
        # Draw the graph
        pos = nx.shell_layout(G) # Position of nodes
        nx.draw_networkx_labels(G, pos, font_size=7, font_color='black', horizontalalignment='center')
        edges_rads = [] # Needed to avoid overlapping edges
        for (u, v, data) in G.edges(data=True):
            rad = 0
            # Manage multiple edges between the same nodes but with different directions
            if (u, v) in multiple_edges and (u, v, 0) in edges_rads or (v, u, 0) in edges_rads:
                rad = 0.2
            # Manage overlapping edges
            tmp_edges_rads = edges_rads.copy()
            while (u, v, rad) in tmp_edges_rads:
                tmp_edges_rads.pop(tmp_edges_rads.index((u, v, rad)))
                rad += 0.2
            # Draw the edge
            color = "black"
            if geometric:
                color = u.split()[0].lower()
                
            nx.draw_networkx_edges(G,
                                pos,
                                edgelist=[(u, v)],
                                width=2,
                                edge_color=color,
                                connectionstyle=f'arc3, rad={rad}',
                                arrowsize = 30,
                                min_source_margin=50,
                                min_target_margin=50)
            nx.draw_networkx_edge_labels(G,
                                        pos,
                                        edge_labels={(u, v): data['label']},
                                        font_color=color,
                                        font_size=22,
                                        font_family="sans-serif",
                                        connectionstyle=f'arc3, rad={rad}')
            edges_rads.append((u, v, rad))
        
        plt.title("Relation Graph")
        
        plt.show()
    
    ####################
    ## INITIALIZATION ##
    ####################
    # Define the f1 score functional
    num_obj = 4 if not geometric else 5
    num_classes = 5 if not geometric else 4

    # Load mu and sigma used to normalize the images
    mu = th.load("./mu")
    std = th.load("./std")

    ######################
    ## MODEL PREDICTION ##
    ######################
    # Generate new image
    test_image_data, relation = create_dataset(1, 128, "./", num_obj, SHAPE_SIZE_MIN, SHAPE_SIZE_MAX, geometric=geometric, rotate=rotate, origami=origami)
    # Load image
    test_image = plt.imread("0000.jpg")

    # Delete the image file
    os.remove("0000.jpg")
    
    # Transform image to tensor
    test_image_tensor = (th.from_numpy(test_image) - mu) / std
    test_image_tensor = test_image_tensor.permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Predict
    model.eval()
    with th.no_grad():
        out, out_relation = model(test_image_tensor)

    ###########################
    ## PREDICTION EVALUATION ##
    ###########################
    out[:, :, :num_classes] = th.softmax(out[:, :, :num_classes], dim=2)

    out = out.cpu().numpy()[0]

    ##########################
    ## VISUALIZATION OUTPUT ##
    ##########################
    plot_output(test_image, test_image_data, out)
    plot_relation(out, out_relation)
