import numpy as np
from collections import OrderedDict
import pandas as pd
import torch
from numpy import math


def s_location():
    M=np.random.randint(0,96,size=12)
    print(M)
    return M
def m_location(x):
    sum = 0
    for i in range(x.size(1)):
        std = x[0][i].std()
        sum = std + sum
    sum=sum/x.size(1)
    sum = sum.cpu()
    sum = sum.detach().numpy()
    if sum>96:
        return math.floor(sum%96)
    if sum<0.95:
        return math.floor(sum*10)
    else:
        return math.floor(sum)