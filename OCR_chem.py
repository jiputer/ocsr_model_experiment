
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
X, Y = parse('/Users/jp/Documents/projects/OCSR_model/dataset') 

#%% splitting data
train_x, val_x, train_y,val_y = train_test_split(X, Y, test_size= 0.10, random_state=69)
train_x = torch.stack(train_x).to(torch.float32) # inputs
train_y = torch.stack(train_y).to(torch.float32)
val_x = torch.stack(val_x).to(torch.float32)
val_y = torch.stack(val_y).to(torch.float32)
train_data = TensorDataset(torch.tensor(train_x).to(torch.float32), torch.tensor(train_y.to(torch.float32)))
test_data = TensorDataset(torch.tensor(val_x).to(torch.float32), torch.tensor(val_y.to(torch.float32)))
loaded_train = DataLoader(train_data, batch_size=128, shuffle=True)
loaded_test = DataLoader(test_data, batch_size=128, shuffle=True)


#%% first attempt, just a typical CNN using LeNet as reference
model = nn.Sequential(
        nn.Conv2d(3, 9, 2, 6),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(9, 18, 2, 3),
        nn.ReLU(),
        nn.MaxPool2d(2,1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3042, 185*len(chems)), 
        nn.Softmax()
)
  
#%% Training and testing functions
def train(loader, model, loss_fn, optimizer):
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        pred_y = model(x)
        # now we have to change the output of y and pred_y

        #print(pred_y.shape, y.shape, x.shape)
        #print(pred_y)
        loss =loss_fn(pred_y, y)
        loss.backward()
        optimizer.step()
        #backwards

def test(loader, model, loss_fn):
    test_acc=0
    loss = 0
    size = 0
    with torch.no_grad():
        #Iterating over the training dataset in batches
        for i, (x, y) in enumerate(loader):
            
            #Calculating outputs for the batch being iterated
            outputs = model(x)
            y_pred = torch.max(outputs)

            #Comparing predicted and true labels
            if (y_pred == y[i]).sum():
                test_acc += 1
            #avg loss
            loss += loss_fn(outputs, y)
            size += 1

    return test_acc, loss, size




#%% training and validating 

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

acc = 0
loss = 0 
size = 0
test_acc = 0
test_loss = 0
test_size = 0
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(loaded_train, model, loss_function, optimizer)
    acc, loss, size = test(loaded_test, model, loss_function)
    test_size += size
    test_acc += acc
    test_loss += loss

# low accuracy here. we need to improve it
print("accuracy: " , test_acc/(test_size), "avg loss:", test_loss/(test_size))
print("Done!")

torch.save(model, "model.pth")
model = torch.load("model.pth")

# %%
