import numpy as np
from numpy import log, exp, zeros, dot, array
import itertools
from itertools import product
import timeit

def log_sum_exp(x):
    
    m = x.max()
    x_s = x - m
    return m + log((exp(x_s)).sum())


def label_seq(l,j):
    '''Creates a list of all posible length l sequences taking values in
       {0,1,...,j-1}. Out put list is of length j^l'''
    
    
    return list(product(list(range(j)), repeat=l))

def find_legit_vals(k,l,n):
    '''Finds the only possible label pairs (i,j) with f_k(i,j,x,t) != 0.
       Will be used to speed up gradient computation.
       l: Number of labels
       n: Size of a nodes feature vector'''
    
    res = []
    k_s = k - n*l
    
    if k_s < 0:  #Means k is in the Unary features
        
        i = k // n
        
        for j in range(l):
            
            res.append((i,j))
            
    else:  #Means k is a Binary feature
        
        f = (k_s // n)
        j = f % l
        i = f // l
        res.append((i,j))  
    
    return res


class CRF(object):
    
    def __init__(self, feature_function, K, T, L, lamb):
        ''' If our labels belong to space L and our observations belonging to X amd
            then length of our chain is T (call {0,1, ... , T-1} = T') then 
            feature_functions is a vector valued function, f: L^2 x X x T' --> R^K
            i.e. f(i,j,x,t) is a K-d real valued vector and has component functions of the form
            f_k(i,j,x,t) to be specified, think of (i,j) = (y_t, y_t-1). It will have an optional
            keyword argument 'project' which defaults to -1, if a positive integer k,
            it will project onto the k-th component.
            
            K: # of Features
            
            T: Length of Chain
            
            L: Number of labels. Will assume labels have been encoded as integers in {0,...,L-1} 
            
            lamb: L2 regularization constant'''
        
        self.f_x = feature_function
        self.W = np.random.randn(K)
        self.K = K
        self.T = T
        self.L = L
        self.Lambda = lamb
        
    def  log_forward(self, x):
        '''This computes the log(alphas) as in the forward-backward algorithm in order to
           be used for inference tasks later on.
           x is an observation.'''
        
        f = self.f_x
        alphas = zeros((self.T, self.L))
        
        # Initialization
        
        for l in range(self.L):
            
            alphas[0,l] = dot(self.W, f(l,0,x,0))
            
        # Recursion
        
        for t in range(1,self.T):
            
            for l in range(self.L):
                
                psi = array([dot(self.W, f(l,i,x,t)) for i in range(self.L)])
                
                alphas[t,l] = log_sum_exp(psi + alphas[t-1])
            
        return alphas
    
        
    def log_backward(self, x):
        '''This computes the log(betas) as in the forward-backward algorithm in order to
           be used for inference tasks later on.
           x is an observation.'''
        
        # Initialization
        
        f = self.f_x
        betas = np.ones((self.T, self.L))
        
        # Recursion
        
        for t in range(self.T-2,-1,-1):
            
            for l in range(self.L):
                
                psi = array([dot(self.W, f(i,l,x,t+1)) for i in range(self.L)])
                
                betas[t][l] = log_sum_exp(psi + betas[t+1])
                
        return betas
    
    
    def log_partition(self, x):
        '''Efficient computation of the log of the partition function Z(x) appearing in CRF model.
           Input an observation and inital label (for forward algorithm) and output is log(Z(x))'''
        
        alphas = self.log_forward(x)
        
        return log_sum_exp(alphas[-1])
    
    
    def MAP(self, x):
        '''Viterbi algortithm for computing the most likely label of a sequence with
           given observation vector x using maximum a posteriori estimation. Using log
           sum version for numeric stability'''
        
        f = self.f_x
            
        # Initialization
        
        deltas = np.zeros((self.T, self.L))
        delt_arg = np.zeros((self.T, self.L))
        
        for l in range(self.L):
            
            deltas[0][l] = dot(self.W, f(l,0,x,0))  # Not sure about this.
            
        # Recursion
        
        for t in range(1,self.T):
            
            for l in range(self.L):
                
                psi = array([dot(self.W, f(l,i,x,t)) for i in range(self.L)])
                
                deltas[t][l] = (psi + deltas[t-1]).max()
                delt_arg[t][l] = (psi + deltas[t-1]).argmax()
            
        map_lab = np.zeros(self.T, dtype='i4')
        map_lab[-1] = deltas[-1].argmax()
        
        for t in range(self.T-2,-1,-1):
            
            map_lab[t] = delt_arg[t+1][map_lab[t+1]]
            
                
        return tuple(map_lab)
        
        
    def marginal(self,i,j,x,t):
        '''Using the forward backward algorithm to compute the marginal p(y_t-1=i,y_t=j|x)'''
        
        f = self.f_x
        alphas = self.log_forward(x)
        betas = self.log_backward(x)
        psi = dot(self.W,f(j,i,x,t))
        psi_b = np.array([dot(self.W,f(k,0,x,0)) for k in range(self.L)])
        log_joint = alphas[t-1][i] + psi + betas[t][j] - log_sum_exp(psi_b + betas[0])
        
        return exp(log_joint)
                       
    def naive_comp(self, x, out='Z'):
        '''Brute force computation of log(Z(x)) or MAP (if out = 'MAP')'''
        
        f = self.f_x
        
        # Get List of all possible label sequences
        
        lab_seq = label_seq(self.T, self.L)
        
        psi = np.zeros(len(lab_seq))
        
        for k in range(len(lab_seq)):
            
            lab = lab_seq[k]
            temp = np.zeros(self.T)
            temp[0] = dot(self.W, f(lab[0], 0, x, 0))
            
            for t in range(1,self.T):
                
                temp[t] = dot(self.W, f(lab[t], lab[t-1], x, t))
            
            psi[k] = temp.sum()
        
        arg_m_i = psi.argmax()
        
        return log_sum_exp(psi) if out == 'Z' else lab_seq[arg_m_i]
        
        
    
    def gradient(self, X, Y):
        ''' Creates the gradient vector of the log-likelihood. 
            X, Y: Are arrays containing training examples.'''
        
        f = self.f_x
        lamb = self.Lambda
        grad = np.zeros(self.K)
        
        for k in range(self.K):
            
            val_pair = find_legit_vals(k, self.L, X.shape[-1])
            first_term = np.zeros((X.shape[0],self.T))
            
            for n in range(X.shape[0]):
                
                for t in range(self.T):
                    
                    first_term[n][t] = f(Y[n][t], Y[n][t-1], X[n], t, project=k)
            
            sec_term = np.zeros((X.shape[0],self.T))
            
            for n in range(X.shape[0]):
                
                for t in range(self.T):
                    
                    marginals = np.zeros(len(val_pair))
                        
                    for j in range(len(val_pair)):
                        
                        y, y_p = val_pair[j]
                        marginals[j] = (f(y, y_p, X[n], t, project=k) * 
                                        self.marginal(y, y_p, X[n], t))
                    
                    sec_term[n][t] = marginals.sum()
                    
            grad[k] = (first_term + sec_term).sum() - self.W[k] * lamb
            print(k)
        return grad




def feat_map(i, j, x, t, projection=-1):
    
    unary_chi = np.zeros(15)
    binary_chi = np.zeros(45)
    
    unary_chi[5*i:5*(i+1)] = (t >=0) * x[t]
    
    if t > 0:
        
        binary_chi[5*(3*i+j): 5*(3*i+j+1)] = x[t] + x[(t-1)]
        
    if t == 0 and j == 0:
        
        binary_chi[5*(3*i+j): 5*(3*i+j+1)] = x[0]
    
    output = np.zeros(3*5 + 9*5)
    output[:3*5] = unary_chi
    output[3*5:] = binary_chi
    
    if projection > 0:
        
        return output[projection]
    
    else:
        
        return binary_chi



features = np.zeros((99, 5, 5))

for k in range(33):
    
    for j in range(5):
        
        features[k,j,0] = np.random.randn()
        features[k+33,j,2] = np.random.randn()
        features[k+66,j,4] = np.random.randn()

labels = np.zeros(99)

for k in range(33):
    
    labels[k+33] = 1
    labels[k+66] = 2



features[0]




crf_test = CRF(feat_map, 45, 5, 3, 1)



crf_test.MAP(features[1])



crf_test.naive_comp(features[1],'MAP')



crf_test.naive_comp(features[0])



crf_test.log_partition(features[0])



m_test = np.array([crf_test.marginal(i,j,features[12],2) for i in range(3) for j in range(3)])



m_test.sum()



from oct2py import octave
octave.addpath('/home/james/anaconda3/data/HW')



X = octave.data_generator()



def feat_map_toy(i, j, x, t, project=-1):
    ''' Change the values of feature vector length and number of classes. '''
    
    N = 4                            # length of data vector
    L = 4                            # number of labels
    T = 10                           # length of sequence
    unary_chi = np.zeros(N * L)
    binary_chi = np.zeros(N * L**2)
    
    unary_chi[N*i:N*(i+1)] = x[t]
    
    if t > 0:
        
        binary_chi[N*(L*i+j): N*(L*i+j+1)] = x[t] + x[(t-1)]
        
    if t == 0 and j == 0:
        
        binary_chi[N*(L*i+j): N*(L*i+j+1)] = x[t]
    
    
    output = np.zeros(N*L + N*L**2)
    output[:N*L] = unary_chi
    output[N*L:] = binary_chi
    
    if project > -1:
        
        return output[project]
    
    else:
        
        return output




crf_test1 = CRF(feat_map_toy,16 + 64,10, 4, 1)



marg_test = np.array([crf_test1.marginal(i,j,X_f[6],3) for i in range(4) for j in range(4)])



marg_test.sum()



X_f = np.zeros((150,10,4))

for k in range(150):
    
    for j in range(10):
        
        X_f[k,j,:] = X[k][0][:,j]



y_f = np.zeros((150,10))

for k in range(150):
    
    y_f[k] = X[k,1][0] -1
    
y_f = np.asarray(y_f,dtype='i4')



y_f



crf_test1.gradient(X_f,y_f)



X.shape[0]



X_f[149]




times, times1 = [], []
for k in range(10):
    
    start = timeit.default_timer()
    test1 = crf_test1.marginal(1,1,X_f[0],2)
    end = timeit.default_timer()
    times.append(end-start)
    
for k in range(10):
    
    start = timeit.default_timer()
    test1 = crf_test1.log_forward(X_f[0])
    end = timeit.default_timer()
    times1.append(end-start)
    
print(0.1*sum(times))
print(0.1*sum(times1))




