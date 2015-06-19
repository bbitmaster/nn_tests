#!/usr/bin/env python
import numpy as np

def Shrink_Cov(X):
    (p,N) = X.shape
    H = np.eye(N) - np.ones((N,N))/float(N)
    S = np.dot(X,H)
    S = np.dot(S,S.T)/float(N - 1)
    Xc = np.dot(X,H)
    N_ = (N/float(((N-1)**2*(N-2))))

    V_ST = N_*( np.sum(np.diag(np.dot((Xc).T,Xc))**2) - ((N-1)**2/float(N))*np.trace(np.dot(S,S.T)))

    V_ST_target = (N_/float(p))*np.sum((np.diag(np.dot(Xc.T,Xc))- ((N-1)/float(N))*np.trace(S))**2)

    ST_target = np.eye(p,p)*np.trace(S)/float(p)

    lambda_ST=(V_ST-V_ST_target)/(np.sum((S-ST_target)**2))

    
    lambda_ST = max(min(lambda_ST,1.0),0.0)
    S_reg = (1 - lambda_ST)*S + lambda_ST*ST_target
    return (S_reg,lambda_ST)

if __name__ == '__main__':
    #X = np.random.random((5,20))

    X = np.array([[1,5,2],
                [2,6,3],
                [3,2,1],
                [4,8,4]])
    (Sx,lambda_ST) = Shrink_Cov(X)
    print("lambda_ST: " + str(lambda_ST))
    print("X: " + str(X))
    print("Sx: " + str(Sx))
    #print(Sx.shape)

    X = np.random.random((768,1024))
    for i in range(50):
        (Sx,lambda_ST) = Shrink_Cov(X)
        print(i)
