# Naive OCSR ?
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
chems = [ "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","K","Ar","Ca","Sc","Ti","V","Cr","Mn","Fe","Ni","Co","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","I","Te","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og","-","+","=", "#", "/", "\\", "[" , "]", "(", ")", "@", "0", "1", "2","3", "4", "5", "6"]
chems_dict = {symbol: index for index, symbol in enumerate(chems)}
# inefficient btw
def chemToArray(chem):
    chem = chem.strip()
    temp = [0] * len(chem)
    for k in chems:
        b = chem.find(k)
        if b > 0:
            a = b
            temp[a] = chems_dict[k]

    return temp

def parse():
    # create a matrix for each entry in the image data set
    # how would i be able to resample these images to create a bigger data set?
    files = os.listdir(path='/Users/jp/Documents/projects/OCSR_model/dataset')
    X = []
    basewidth = 400
    max_formula_len = 185

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
        chemical = chemToArray(l) # change this into an array of chemicals at position
        chemical_len = len(chemical)
        if chemical_len < max_formula_len:
            # add an array of abs(chemical_len - max_formula_len)
            chemical += [0] * (max_formula_len - chemical_len)
        elif chemical_len > max_formula_len:
            print(f"there is a chemical formula that's longer than the {max_formula_len}")
        # new chemical
        chemical = torch.Tensor(chemical)
        Y.append(chemical)
    return X, Y

# split the data for training and validation
X, Y = parse()
x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.25, random_state=69)




# %% embeddings (integer look up)



# %% positional encodings???? do we need it? uhhh sure ig.




# %% training the model

def forward_pass():
    pass 

# %% using the model to predict

# standarized the image to the training size

def predict():
    pass
