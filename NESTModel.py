import numpy as np

def NEST_DL(F, A=57.38, APow=-0.12221, B=127.27, BPow=32.821):
    par1 = A*np.power(F, APow)
    par2 = B*(np.exp(-F/BPow))
    return par1 + par2

def NEST_DV(x, A=-1.77599343e+00, B=2.02409698e+01, C=3.85313848e-03):
    return np.exp(A - B/x + C*np.log10(x))