#!/usr/bin/env python3
import numpy as np


def hnl1(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return np.pi**(x0*x1)*(2*np.abs(x2))**(1/2) - np.arcsin(x3) + np.log(np.abs(x2+x4)) - x8/(x9+eps)*np.abs(x6/(x7+eps))**(1/2) - x1*x6

def hnl2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return np.pi**(x0*x1)*(2*np.abs(x2))**(1/2) - np.arcsin(0.5*x3) + np.log(np.abs(x2+x4) + 1) - x8/(1+np.abs(x9))*np.abs(x6/(1+np.abs(x7)))**(1/2) - x1*x6

def hnl3(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return np.exp(np.abs(x0 - x1)) + np.abs(x1*x2) - np.abs(x2)**(2*np.abs(x3)) + np.log(x3**2 + x4**2 + x6**2 + x7**2) + x8 + 1/(1+x9**2)

def hnl4(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return np.exp(np.abs(x0 - x1)) + np.abs(x1*x2) - np.abs(x2)**(2*np.abs(x3)) + np.log(x3**2 + x4**2 + x6**2 + x7**2) + x8 + 1/(1+x9**2) + x0**2*x3**2

def hnl5(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return 1/(1+x0**2+x1**2+x2**2) + np.exp(x3+x4)**(1/2) + np.abs(x5+x6) + x7*x8*x9

def hnl6(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return np.exp(np.abs(x0*x1+1)) - np.exp(np.abs(x2+x3)+1) + np.cos(x4+x5-x7) + (x7**2+x8**2+x9**2)**(1/2)

def hnl7(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return (np.arctan(x0) + np.arctan(x1))**2 + np.maximum(x2*x3 + x5, 0) - 1/(1+(x3*x4*x5*x6*x7)**2) + (np.abs(x6)/(1+np.abs(x8)))**5 + (x0+x1+x2+x3+x4+x5+x6+x7+x8+x9)

def hnl8(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return x0*x1 + 2**(x2+x4+x5) + 2**(x2+x3+x4+x6) + np.sin(x6*np.sin(x7+x8)) + np.arccos(0.9*x9)

def hnl9(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return np.tanh(x0*x1 + x2*x3)*abs(x4)**2 + np.exp(x4+x5) + np.log(x5**2*x6**2*x7**2+1) + x8*x9 + 1/(1+np.abs(x9))

def hnl10(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return np.sinh(x1+x2) + np.arccos(np.tanh(x2+x4+x6)) + np.cos(x3+x4) + 1/np.cos(x6*x8)

def nl1(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return x0 + 5*x1 + x1*x2 + x3**2*x5 + x4 + x8*x9

def nl2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return np.abs(x0)**x1 + x2*np.abs(x3)**4 + (3-x5)**x6 + 6*x8

def nl3(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return (x1*x2*np.abs(x3*x4))**3 + (x1*x6*np.abs(x7*x8))**3 + x0 + x9**5

def nl4(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return np.abs(x1*x2*np.abs(x3*x4))**x3 + np.abs(x1*x6*np.abs(x7*x8))**x5 + x0 + x9**5

def nl5(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, eps=1e-6):
    return x1 + np.cos(x1*x2*np.abs(x3*x4)) + (x1*x6*np.abs(x7*x8))**3 + x0 + x9**5


def generate_data(n_samples, function, random_state=None):
    np.random.seed(random_state)
    X = np.random.uniform(-1, 1, (n_samples, 10))
    y = function(X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6],X[:,7],X[:,8],X[:,9])
    np.random.seed(None)
    return X, y

