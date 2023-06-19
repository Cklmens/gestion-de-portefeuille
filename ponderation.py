import numpy as np
import math as ma
import matplotlib.pyplot as plt

def weight(n):
    const= 5
    Wi=np.zeros(n)
    Wi=np.random.random(low=0.0, high=1.0)
    for i in range(const):
        for j in range(n):
          Wi[i][j]=Wi[i][j] /np.sum(Wi[i])
    return Wi





    
    
