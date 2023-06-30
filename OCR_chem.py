# Naive OCSR ?
#%%
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, Sequential
import torchvision.transforms as transforms
import pandas as pd
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt


# %%

chems = [ "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","K","Ar","Ca","Sc","Ti","V","Cr","Mn","Fe","Ni","Co","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","I","Te","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og","-","+","=", "#", "/", "\\", "[" , "]", "(", ")", "@", "0", "1", "2","3", "4", "5", "6"]
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
def parse(folder_path):
    # create a matrix for each entry in the image data set
    # how would i be able to resample these images to create a bigger data set?
    files = os.listdir(path=folder_path)
    X = []
    size = 500
    max_formula_len = 185

    for file in files:
        img = Image.open(f'dataset/{file}')
        #img = img.convert(mode="1", dither=Image.Dither.FLOYDSTEINBERG)
        img = img.resize((size,size), Image.Resampling.LANCZOS) # squeeze into a 500 by 500 thing.
        transform = transforms.Compose([transforms.PILToTensor()])
        d = torch.squeeze(transform(img).to(torch.uint8))
        X.append(d)
    
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


#%% 
X, Y = parse('/Users/jp/Documents/projects/OCSR_model/dataset') 

#%%
train_x, val_x, train_y,val_y = train_test_split(X, Y, test_size= 0.10, random_state=69)
train_x = torch.stack(train_x) # inputs
train_y = torch.stack(train_y)
val_x = torch.stack(val_x)
val_y = torch.stack(val_y)


#%% cheating using pytorch (for now)
model = Sequential(
    Conv2d(64, 64, kernel_size=3, stride=3),
    MaxPool2d(3, 2),
    Conv2d(in_channels=64, out_channels=64, kernel_size=4),
    MaxPool2d(3, 2),
    ReLU()
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

train_data = TensorDataset(torch.tensor(train_x).to(torch.float32), torch.tensor(train_y.to(torch.float32)))
test_data = TensorDataset(torch.tensor(val_x).to(torch.float32), torch.tensor(val_y.to(torch.float32)))
loaded_train = DataLoader(train_data, batch_size=5, shuffle=True)
loaded_test = DataLoader(test_data, batch_size=5, shuffle=True)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
       
        pred = model(X.to(torch.float32))
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            #print(X)
            pred = model(X.to(torch.float32))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(loaded_train, model, loss_function, optimizer)
    test(loaded_test, model, loss_function)
print("Done!")

torch.save(model, "model.pth")
model = torch.load("model.pth")
