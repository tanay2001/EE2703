"""
EE2703 assignment 3
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: Feb 28, 2021

"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from pylab import errorbar
import pylab
import pandas as pd 
from scipy import linalg
import scipy.special as sp
import os
os.chdir("/home/tanay/Documents/sem4/EE2703/week3/")

#f(t) = 1.05J2(t) - 0.105t + n(t)
def plot_function(x,y,sigma,path):
    '''
    saves the polt as a .png file in path
    Params:
    path : image path 
    x is 1 dim time axis
    y is 9 dim fucntions values
    returns None
    '''

    #sigma = np.logspace(-1,-3,9)
    for i, error in enumerate(sigma):
        plt.plot(x, y[:,i], label ='$\sigma$ = %.3f'%error)

    true = g(x,1.05,-0.105)
    plt.plot(x,true, color ='k', label ='True Value')
    plt.legend()
    plt.savefig(path+'.png',bbox_inches='tight')
    print("file saved")
    plt.clf()


def g(t,A,B):
    return A*sp.jn(2,t)+ B*t


def errorsplot(t, data, sigma,path):
    errorbar(t[::5],data[::5,1],sigma[0],fmt='ro',label = "Error bar")
    plt.xlabel("t")
    plt.title("Data points for $\sigma$ = 0.1 along with exact function")
    plt.plot(t,g(t,1.05,-0.105),label = "True value")
    plt.legend()
    plt.savefig(path+'.png',bbox_inches='tight')
    print('file saved')
    plt.clf()


def fillM(x):
    M = np.zeros((x.shape[0],2))
    M[:,0] = sp.jn(2,x)
    M[:,1] = x
    return M
    
def check(x):
    p = np.array([1.05,-0.105])
    M = fillM(x)
    pred = np.matmul(M,p)
    true = np.array(g(x,1.05,-0.105))

    return np.array_equal(pred, true)

def ls_estimate(M,p):
    try:
        return linalg.lstsq(M,p)[0]
    except:
        print("could not esitmated model params for a case")
        exit()


def errorMatrix(x,y,col):

    yt = y[:,col]
    error = np.empty((21,21))
    A = np.linspace(0,2,21)
    B = np.linspace(-0.2,0,21)
    for i in range(21):
        for j in range(21):
            error[i,j] = np.mean(np.square(g(x,A[i],B[j]) - yt))
    return error

def contourplot(x,y, path):
    a = np.linspace(0,2,21)
    b = np.linspace(-0.2,0,21)
    A, B = np.meshgrid(a,b)
    error = errorMatrix(x,y,0)
    minimum = np.argmin(error)
    annot = np.unravel_index(minimum,error.shape)
    CS = pylab.contour(A,B,error,np.linspace(0.025, 0.5 ,20))
    plt.clabel(CS,CS.levels[:4], inline=1, fontsize=8)
    pylab.annotate('(%0.3f,%0.3f)'%(a[annot[0]],b[annot[1]]), (a[annot[0]],b[annot[1]]))
    plt.title('Contour Plot')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.savefig(path+'.png',bbox_inches='tight')
    print("file saved")
    plt.clf()

def linearplot(sigma, Aerr, Berr, path):
    pylab.plot(sigma,Aerr,'bo',label='Aerr')
    pylab.plot(sigma,Berr,'ro',label='Berr')
    plt.xlabel("Noise standard deviation ->")
    plt.ylabel('MS error')
    plt.title('Q10: Variation of error with noise')
    plt.savefig(path+'.png',bbox_inches='tight')
    print("file saved")
    plt.clf()


def logplot(sigma , Aerr, Berr, path):
    pylab.loglog(sigma,Aerr,'ro')
    pylab.stem(sigma,Aerr,'-ro', use_line_collection=True)
    pylab.loglog(sigma,Berr,'bo')
    pylab.stem(sigma,(Berr),'-bo', use_line_collection=True)
    pylab.xlabel(r'$\sigma_{n}\rightarrow$')
    pylab.ylabel(r'MSerror$\rightarrow$')
    plt.savefig(path+'.png',bbox_inches='tight')
    print("file saved")
    plt.clf()



if __name__ =='__main__':

    data = np.loadtxt('fitting.dat',dtype=float)
    x  = np.array(data[:,0])
    y = np.asarray(data)[:,1:]
    plot_function(x,y, np.logspace(-1,-3,9),'figure0')
    errorsplot(x,y, np.logspace(-1,-3,9),'figure1')

    contourplot(x,y,'figure2')
    estimates =[]
    M = fillM(x)
    for i in range(9):
        estimates.append(ls_estimate(M,y[:,i]))

    e = np.asarray(estimates)

    A_error = np.square(e[:,0] -1.05)
    B_error = np.square(e[:,1] + 0.105)

    sigma = np.logspace(-1,-3,9)


    linearplot(sigma, A_error, B_error, 'linearplot')

    logplot(sigma, A_error, B_error, 'logplot')
    
























