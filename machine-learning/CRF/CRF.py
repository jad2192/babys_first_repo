import numpy as np
from numpy import log, exp, zeros, dot, array
import itertools
from itertools import combinations_with_replacement as CWR
from itertools import permutations


def log_sum_exp(x):
    
    m = x.max()
    x_s = x - m
    return m + log((exp(x_s)).sum())


def label_seq(l,j):
    '''Creates a list of all posible length l sequences taking values in
       {0,1,...,j-1}. Output list is of length j^l'''
    
    return list(set(CWR(range(j),l)).union(set(itertools.permutations(range(j),l))))


class CRF(object):
    
    def __init__(self, feature_functions, K, T, L):
        ''' If our labels belong to space L and our observations belonging to X amd
            then length of our chain is T (call {0,1, ... , T-1} = T') then 
            feature_functions is a vector valued function, f: L^2 x X x T' --> R^K
            i.e. f(i,j,x,t) is a K-d real valued vector and has component functions of the form
            f_k(i,j,x,t) to be specified, think of (i,j) = (y_t, y_t-1).
            
            K: # of Features
            
            T: Length of Chain
            
            L: Number of labels. Will assume labels have been encoded as integers in {0,...,L-1} '''
        
        self.f_x = feature_function
        self.W = np.random.randn(K)
        self.K = K
        self.T = T
        self.L = L
        
        
    def  log_forward(self, x, y_0):
        '''This computes the log(alphas) as in the forward-backward algorithm in order to
           be used for inference tasks later on.
           x is an observation and y_0 is intial state.'''
        
        f = self.f_x
        alphas = zeros((self.T, self.L))
        
        # Initialization
        
        for l in range(self.L):
            
            alphas[0][l] = dot(self.W, f(l,y_0,x,1))
            
        # Recursion
        
        for t in range(1,self.T):
            
            for l in range(self.L):
                
                psi = array([dot(self.W, f(l,i,x,t+1)) for i in range(self.L)])
                
                alphas[t][l] = log_sum_exp(psi + alphas[t-1])
            
        return alphas
    
        
    def log_backward(self, x):
        '''This computes the log(betas) as in the forward-backward algorithm in order to
           be used for inference tasks later on.
           x is an observation.'''
        
        # Initialization
        
        f = self.f_x
        betas = np.zeros((self.T, self.L))
        
        # Recursion
        
        for t in range(self.T-2,-1,-1):
            
            for l in range(self.L):
                
                psi = array([dot(self.W, f(i,l,x,t)) for i in range(self.L)])
                
                betas[t][l] = log_sum_exp(psi + betas[t+1])
                
        return betas
    
    
    def log_partition(self, x, y_0):
        '''Efficient computation of the log of the partition function Z(x) appearing in CRF model.
           Input an observation and inital label (for forward algorithm) and output is log(Z(x))'''
        
        alphas = self.log_forward(x, y_0)
        
        return log_sum_exp(alphas[-1])
    
    
    def MAP(self, x):
        '''Viterbi algortithm for computing the most likely label of a sequence with
           given observation vector x using maximum a posteriori estimation. Using log
           sum version for numeric stability'''
        
        f = self.f_x
            
        # Initialization
        
        deltas = np.zeros((self.T, self.L))
        delt_arg = np.zeros(self.T, self.L)
        
        for l in range(self.L):
            
            deltas[0][l] = dot(self.W, f(l,y_0,x,1))  # Not sure about this
            
        # Recursion
        
        for t in range(1,self.T):
            
            for l in range(self.L):
                
                psi = array([dot(self.W, f(l,i,x,t+1)) for i in range(self.L)])
                
                deltas[t][l] = (psi + deltas[t-1]).max()
                delt_arg[t][l] = (psi + deltas[t-1]).argmax()
            
        map_lab = np.zeros(self.L)
        map_lab[-1] = deltas[-1].argmax()
        
        for t in range(self.L-1,-1,-1):
            
            map_lab[t] = delt_arg[t+1][map_lab[t+1]]
            
                
        return map_lab
        
                       
    def naive_comp(self, x, out='Z'):
        '''Brute force computation of log(Z(x)) or MAP (if out = 'MAP')'''
        
        f = self.f_x
        
        # Get List of all possible label sequences
        
        lab_seq = label_seq(self.T, self.L)
        
        psi = np.zeros(len(lab_seq))
        
        for k in range(len(lab_seq)):
            
            lab = lab_seq[k]
            temp = np.zeros(self.T)
            
            for t in range(1,self.T):
                
                temp[t] = dot(self.W, f(lab[t], lab[t-1], x, t))
            
            psi[k] = temp.sum()
        
        arg_m_i = psi.argmax()
        
        return log_sum_exp(psi) if out == 'Z' else lab_seq[arg_m_i]
        

        




