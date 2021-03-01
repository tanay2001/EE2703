"""
EE2703 assignment 3
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: Feb 28, 2021

"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from pylab import errorbar
import pandas as pd 
from scipy import linalg
import scipy.special as sp

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
        plt.plot(x, y['example{}'.format(i)], label ='$\sigma_{}$ = %.3f'.format(i+1)%error)

    true = g(x,1.05,-0.105)
    plt.plot(x,true, color ='k', label ='True Value')
    plt.legend()

    
    plt.savefig(path+'.png',bbox_inches='tight')
    plt.clf()


def g(t,A,B):
    return A*sp.jn(2,t)+ B*t


def errorsplot(t, data, sigma,path):
    errorbar(t[::5],data[::5,1],sigma[1],fmt='ro',label = "Error bar")
    plt.xlabel("t",size=20)
    plt.title("Q5:Data points for $\sigma$ = 0.1 along with exact function")
    plt.plot(t,g(t,1.05,-0.105),label = "True value")
    plt.legend()
    plt.savefig(path+'.png',bbox_inches='tight')
    plt.clf()


def fillM(x):
    M = np.zeros((x.shape[0],2))
    M[:,0] = sp.jn(2,x)
    M[:,1] = x
    return M
    
def generateAB(i,j,step1 = 0.1,step2 = 0.01,Amin=0,Bmin = -0.2):
    p = np.zeros((2,1))
    p[0][0] = Amin +  step1 * i
    p[1][0] = Bmin +step2 * j
    return p


def find_error_matrix(x,y,col):

    yt = np.reshape(y[:,col],(101,1))
    error = np.zeros((20,20))
    M = fillM(x)
    for i in range(20):
        for j in range(20):
            error[i,j] = np.square( np.matmul(M,generateAB(i,j)) - yt).mean()
    return error

def contourplot(x,y, path):
    a = np.linspace(0,2,20)
    b = np.linspace(-0.2,0,20)
    X, Y = np.meshgrid(a,b)
    error = find_error_matrix(x,y,0)
    CS = plt.contour(X,Y,error,np.linspace(0.025, 0.5 ,20))
    plt.clabel(CS,CS.levels[:4], inline=1, fontsize=10)
    plt.title('Contour Plot of error')
    plt.xlabel(r'$A$',size=10)
    plt.ylabel(r'$B$',size=10)
    plt.savefig(path+'.png',bbox_inches='tight')
    plt.clf()


def find(M,b):
    return linalg.lstsq(M,b)

def error_pred(pred,true):
    return np.square(pred[0]-true[0]),np.square(pred[1]-true[1])




if __name__ =='__main__':

    data = pd.read_csv('/home/tanay/Documents/sem4/EE2703/week3/fitting.dat', delimiter=" ", names =['time']+['example{}'.format(i) for i in range(9)])
    #plot_function(data['time'], data.loc[:,'example0':'example8'], np.logspace(-1,-3,9),'/home/tanay/Documents/sem4/EE2703/week3/figure0')
    #errorsplot(data['time'].values, data.loc[:,'example0':'example8'].values, np.logspace(-1,-3,9),'/home/tanay/Documents/sem4/EE2703/week3/figure1')

    #A,B,e = mse(data['time'].values)
    contourplot(data['time'].values, data.loc[:,'example0':'example8'].values,'/home/tanay/Documents/sem4/EE2703/week3/figure2')
    AB = np.zeros((2,1))
    x = data['time'].values
    y = data.loc[:,'example0':'example8'].values
    AB[0][0] = 1.05
    AB[1][0] = -0.105
    scl=np.logspace(-1,-3,9)
    error_a = np.zeros(9)
    error_b = np.zeros(9)
    error_c = np.zeros(9)
    for i in range(9):
        prediction,error,_,_ = find(fillM(x),y[:,i])
        error_a[i],error_b[i] = error_pred(prediction,AB)
        error_c[i] = error


    plt.plot(scl,error_a,'r--')
    plt.scatter(scl,error_a)
    plt.plot(scl,error_b, 'b--')
    plt.scatter(scl,error_b)
    plt.legend(["A","B"])
    plt.title("Variation Of error with Noise")
    plt.xlabel('$\sigma_n$',size=10)
    plt.ylabel('MS Error',size=10)
    path = '/home/tanay/Documents/sem4/EE2703/week3/figure3'
    plt.savefig(path+'.png',bbox_inches='tight')
    plt.clf()

    plt.loglog(scl,error_a,'r--',basex = 10)
    plt.scatter(scl,error_a)
    plt.loglog(scl,error_b, 'b--',basex = 10)
    plt.scatter(scl,error_b)
    plt.legend(["A","B"])
    plt.title("Variation Of error with Noise on loglog scale")
    plt.xlabel('$\sigma_n$',size=10)
    plt.ylabel('MS Error',size=10)
    path = '/home/tanay/Documents/sem4/EE2703/week3/figure4'
    plt.savefig(path+'.png',bbox_inches='tight')
    plt.clf()

    plt.loglog(scl,error_c, 'b--',basex = 10)
    plt.scatter(scl,error_c)
    plt.title("Variation Of error returned by Lstsq with Noise on loglog scale")
    plt.xlabel('$\sigma_n$',size=10)
    plt.ylabel('MS Error',size=10)
    path = '/home/tanay/Documents/sem4/EE2703/week3/figure5'
    plt.savefig(path+'.png',bbox_inches='tight')
    plt.clf()
























