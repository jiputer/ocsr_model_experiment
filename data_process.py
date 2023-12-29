
#%%
from PIL import Image
from PIL import ImageOps
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib.pyplot as plt

'''
Data processing ideas:

find the minimum darkness value and if the value is less than or equal to this darkness value
in the image. Set it to zero if under the threshold
I could just do this inside the model (?)

'''
# %%
#chems = [ "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","K","Ar","Ca","Sc","Ti","V","Cr","Mn","Fe","Ni","Co","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","I","Te","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og","-","+","=", "#", "/", "\\", "[" , "]", "(", ")", "@", "0", "1", "2","3", "4", "5", "6"]
# DECIMER SET BASED SET
chems = ["C", "H", "O", "N", "P", "S", "F", "Cl", "Br", "I", "Se", "B", "-","+","=", "#", "/",  "[" , "]", "(", ")", "@", "0", "1", "2","3", "4", "5", "6"]

chems_dict = {symbol: index for index, symbol in enumerate(chems)}
def arrayToChem(arr):
    j = ""
    # all values in arr must be a int
    for i in arr:
        j += chems[i]
    return j

def chemToArray(chem):
    chem = chem.strip()
    temp = [0] * len(chem)
    for k in chems:
        b = chem.find(k)
        if b > 0:
            a = b
            temp[a] = chems_dict[k]

    return temp
#%%
# Scuffed parsing; inefficient takes about 15mins XDDDDDD
def parse(folder_path):
    # 1) takes all image in the folder
    # 2) converts it to the same size to around the same ratio as the max

    files = os.listdir(path=folder_path)
    X = []
    # converted_X = []

    max_height , max_width = 0 , 0
    max_formula_len = 185 #196 so i can square it to 14 x 14 maybe?
    transform = transforms.Compose([transforms.PILToTensor()])

    # Square work around
    for file in files:
        img = Image.open(f'dataset/{file}')
        img = img.convert(mode="RGB")
        img = img.resize((500,500))
        d = torch.squeeze(transform(img).to(torch.uint8))
        X.append(d)

    # Asymmetric work  around

    # look at each file and convert them to a certain size
    # for file in files:
    #     img = Image.open(f'dataset/{file}')
    #     img = img.convert(mode="RGB")
        
    #     X.append(img)
                # reset max_width and max_height

    # for v in X:
    #     v = ImageOps.pad(v, (max_width, max_height), color=254) #pad each image first
    #     v = img.resize((500,500)) # resize; some of them could be very small so that can be an issue
    #     d = torch.squeeze(transform(v).to(torch.uint8))
    #     converted_X.append(d)
    # X = converted_X

    
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