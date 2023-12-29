# Feat
'''
Data processing ideas:

find the minimum darkness value and if the value is less than or equal to this darkness value
in the image. Set it to zero if under the threshold
I could just do this inside the model (?)

'''
import torch
import torch.nn as nn

# TODO
# 1) design the model
# 2) signal increasing?
# 3) what else do i need to do???
#

# idea 

# pass CNN -> LSTM -> generate a string
'''
To train 

'''


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        
        self.convDown = nn.Conv2d()


class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        up1 = UpSample()
        down1 = DownSample()