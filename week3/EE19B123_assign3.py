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


#f(t) = 1.05J2(t) - 0.105t + n(t)
def plot_function(x,y,sigma,path):
    '''
    plots the various function values for different noise values
    path : image path 
    x is 1 dim time axis
    y is 9 dim fucntions values
    returns None
    please give suitable file path for svaing the image as .png and uncomment the lines 38 and 39
    '''

    #sigma = np.logspace(-1,-3,9)
    for i, error in enumerate(sigma):
        plt.plot(x, y[:,i], label ='$\sigma$ = %.3f'%error)

    true = g(x,1.05,-0.105)
    plt.plot(x,true, color ='k', label ='True Value')
    plt.xlabel('t')
    plt.ylabel('f(t)+ noise')
    plt.title("Data to be fitted to theory")
    plt.legend()
    #plt.savefig(path+'.png',bbox_inches='tight')
    #print("file saved")
    plt.clf()


def g(t,A,B):
    return A*sp.jn(2,t)+ B*t


def errorsplot(t, data, sigma,path):
    errorbar(t[::5],data[::5,1],sigma[0],fmt='ro',label = "Error bar")
    plt.xlabel("t")
    plt.ylabel('f(t)')
    plt.title("Data points for $\sigma$ = 0.1 along with exact function")
    plt.plot(t,g(t,1.05,-0.105),label = "True value")
    plt.legend()
    #plt.savefig(path+'.png',bbox_inches='tight')
    #print('file saved')
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
        print("could not esitmate model params for a case")
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
    '''
    plots the error matrix  contour plot
    x : time array
    y : the noisy function values

    please give suitable file path and uncomment the lines 112 and 113 for saving image as .png file
    '''
    a = np.linspace(0,2,21)
    b = np.linspace(-0.2,0,21)
    A, B = np.meshgrid(a,b)
    #get error matrix
    error = errorMatrix(x,y,0)
    #find minima
    minimum = np.argmin(error)
    annot = np.unravel_index(minimum,error.shape)
    CS = pylab.contour(A,B,error,np.linspace(0.025, 0.5 ,20))
    plt.clabel(CS,CS.levels[:4], inline=1, fontsize=8)
    pylab.annotate('(%0.3f,%0.3f)'%(a[annot[0]],b[annot[1]]), (a[annot[0]],b[annot[1]]))
    plt.title('Contour Plot of $\epsilon_{ij}$')
    plt.xlabel('A')
    plt.ylabel('B')
    #plt.savefig(path+'.png',bbox_inches='tight')
    #print("file saved")
    plt.clf()

def linearplot(sigma, Aerr, Berr, path):
    '''
    plots the error v/s sigma on lineae scale

    Aerr : array of error in estimates in A
    Berr : array of error in estimated in B 

    please give suitable file path and uncomment the lines 133 and 134 for saving image as .png file
    '''
    pylab.plot(sigma,Aerr,'bo',label='Aerror')
    pylab.plot(sigma,Berr,'ro',label='Berror')
    plt.xlabel("Noise standard deviation ->")
    plt.ylabel('error')
    plt.title('Variation of error with noise')
    plt.legend()
    plt.savefig(path+'.png',bbox_inches='tight')
    print("file saved")
    plt.clf()


def logplot(sigma , Aerr, Berr, path):
    '''
    plots the error v/s sigma on log scale

    Aerr : array of error in estimates in A
    Berr : array of error in estimated in B 

    please give suitable file path and uncomment the lines 155 and 156 for saving image as .png file
    '''
    pylab.loglog(sigma,Aerr,'ro', label ='Aerror')
    pylab.stem(sigma,Aerr,'-ro', use_line_collection=True)
    pylab.loglog(sigma,Berr,'bo', label = 'Berror')
    pylab.stem(sigma,(Berr),'-bo', use_line_collection=True)
    plt.title('Variation of error with noise')
    pylab.xlabel('$\sigma_{n}$ in log scale')
    pylab.ylabel('error(log scale)')
    plt.legend()
    plt.savefig(path+'.png',bbox_inches='tight')
    print("file saved")
    plt.clf()



if __name__ =='__main__':
    ###reading the data file
    try:
        data = np.loadtxt('fitting.dat',dtype=float)
    except FileNotFoundError:
        print("file not found")

    x  = np.array(data[:,0]) ##time
    y = np.asarray(data)[:,1:] ##function values

    #Q3 & Q4
    plot_function(x,y, np.logspace(-1,-3,9),'Q1_main')  ## last param is file path to save image 
    #Q5
    errorsplot(x,y, np.logspace(-1,-3,9),'Q2_error')    ## last param is file path to save image
    #Q6
    assert check(x), "failed test 6"
    #Q8
    contourplot(x,y,'contour')


    #computing errors for all possible values of sigma
    estimates =[]
    M = fillM(x)
    for i in range(9):
        estimates.append(ls_estimate(M,y[:,i]))

    e = np.asarray(estimates)

    A_error = np.abs(e[:,0] -1.05) # error in estimation of A
    B_error = np.abs(e[:,1] + 0.105) # error in estimation of B 

    sigma = np.logspace(-1,-3,9)

    #Q10
    linearplot(sigma, A_error, B_error, 'linearplot') ## last param is file path to save image
    #Q11
    logplot(sigma, A_error, B_error, 'logplot')  ## last param is file path to save image
    
























