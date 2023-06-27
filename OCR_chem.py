#building my own neural net layers

#%%
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import pandas as pd
import os
import jax.numpy as jnp
from jax import grad, jit, vmap, random, image
import matplotlib.pyplot as plt


# %%
# parsing the input with the labels
# make other samples of the input to train with (how should we do this)
# setting the model up
def parse():
    # create a matrix for each entry in the image data set
    # how would i be able to resample these images to create a bigger data set?
    files = os.listdir(path='/Users/jp/Documents/projects/OCSR_model/dataset')
    X = []
    basewidth = 400
    max_formula_len = 250

    for file in files:
        img = Image.open(f'dataset/{file}')
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
        transform = transforms.Compose([transforms.PILToTensor()])
        X.append(transform(img))
        
    
    labels = pd.read_csv('DECIMER_HDM_Dataset_SMILES.tsv', sep='\t', header=0)
    labels = labels['SMILES'].values
    Y = []
    for l in labels:
        chemical = [ord(c) for c in l]
        chemical_len = len(chemical)
        if chemical_len < max_formula_len:
            # add an array of abs(chemical_len - max_formula_len)
            chemical += [0] * (max_formula_len - chemical_len)
        elif chemical_len > max_formula_len:
            print(f"there is a chemical formula that's longer than the {max_formula_len}")
            
        Y.append(chemical)
    
    # we need to create a big size; add 0 entries in the lists to the maximum

    
    
    # label matrix should be 
    #labels = torch.tensor(labels['SMILES'].values)
    # parse characters to ascii

    return X, Y

# split the data for training and validation
X, Y = parse()

# %% embeddings (integer look up)



# %% positional encodings???? do we need it?


# %% training the model
x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.25, random_state=69)

def train():
    pass 

# %% using the model to predict

# standarized the image to the training size

def predict():
    pass
