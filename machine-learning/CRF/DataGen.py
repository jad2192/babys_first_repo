import numpy as np
from numpy import random as rnd
import cv2

def gen_data(N=100, L=8, NData=160, NRows=1, Label_Hsize=50, Label_sigma=7, 
             Noise_sigma=4 ):
    
    Label_Hsize += (1 - (Label_Hsize % 2 ))
    Labels = np.zeros((NData, N))
    Features = np.zeros((NData, NRows, N, L))
    
    for i in range(NData):
        
        tmpM = np.zeros((NRows, N, L))
        
        for l in range(L):
            
            tmpM[:,:,l] = cv2.GaussianBlur(rnd.uniform(low=1e-8,size=(NRows,N)),
                                           ksize=(Label_Hsize, Label_Hsize),
                                           sigmaX=Label_sigma, borderType=cv2.BORDER_REPLICATE)
            
        Labels[i] = np.argmax(tmpM, axis=2)
    
    for i in range(NData):
        
        tmpM = np.zeros((NRows, N, L))
        
        for l in range(L):
            
            tmpM[:,:,l] = int((Labels[i][l] == l))
            
        noise = Noise_sigma * rnd.randn(NRows, N, L)
        Features[i] = tmpM + noise
        
    return Features, Labels
        
