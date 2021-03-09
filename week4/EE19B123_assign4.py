'''
EE2703 assignment 4
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: March 7, 2021

'''

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import quad 
from scipy.linalg import lstsq
import os

#################
# creating a directory to store images 
if os.path.isdir('imgs'):
    pass
else:
    os.mkdir('imgs')
##################

def exponential(x):
    return np.exp(x)

def coscos(x):
    return np.cos(np.cos(x))


def plotdata(x,y1,y2 =[] ,xname= None, yname = None ,icon = 'ro',label = 'plot', path  = None, plottype = None, clear  = True):
    '''
    General plotting function that takes in various params and saves the plot as .png file 

    Params:
    x,y1,y2 are the data columns to be plotted ,y2 (optional)
    xname, yname are name to be given to axis, (optional)
    icon is marker icon (default is red dots)
    label : name given to graph (default : plot)
    path : place to save the image (required)
    plottype: The type pf plot to be use ie: semilogy, loglog, plot (required)
    '''
    if len(y2) >0:
        for i,y in enumerate([y1,y2]):
            assert len(x) ==len(y), "fucntions not same length"
            if plottype =='semilogy': #TODO ad more plot types
                plt.semilogy(x,y,icon[i] ,label =label[i], ms =4)
            elif plottype =='loglog':
                plt.loglog(x,y,icon[i] ,label =label[i],ms =4) 
            else:
                plt.plot(x,y,'r-', label = label[i])
    else:
        assert len(x) ==len(y1), "fucntions not same length"
        if plottype =='semilogy': #TODO ad more plot types
            plt.semilogy(x,y1,icon ,label =label, ms =4)
        elif plottype =='loglog':
            plt.loglog(x,y1,icon ,label =label,ms =4) 
        else:
            plt.plot(x,y1,'r-', label = label)
        
    plt.grid(True)
    plt.legend(loc ='upper right')
    plt.xlabel(xname) 
    plt.ylabel(yname)
    if path !=None:
        plt.savefig(path+'.png',bbox_inches='tight')
        print("file saved at {}".format(path+'.png'))
    if clear:
        plt.clf()

def fourier_coeff(n,func):
    '''
    Params
    n: is the order up-to which to generate fourier series coefficients
    func : is the function whose fourier coefficients are to be found 
    '''
    coeff = np.empty(n)
    u = lambda x,k: func(x)*np.cos(k*x)
    v = lambda x,k: func(x)*np.sin(k*x)
    coeff[0]= quad(func,0,2*np.pi)[0]/(2*np.pi)
    for i in range(1,n,2): 
        coeff[i] = quad(u,0,2*np.pi,args=((i+1)/2))[0]/np.pi
    for i in range(2,n,2):
        coeff[i] = quad(v,0,2*np.pi,args=(i/2))[0]/np.pi
    return coeff

def leastSquareCoef(func):
    '''
    fucntion used to compute the A , b matrixes for Least Sqaure Estimate case 
    '''
    A = np.empty((400,51))
    x = np.linspace(0,2*np.pi, 401)
    x = x[:-1]
    A[:,0] =1 
    for i in range(1,26):
        A[:,2*i-1] = np.cos(i*x)
        A[:,2*i] = np.sin(i*x)
    b = func(x)

    return A, b
def compare(coef1, coef2):
    '''
    computes the maximun deviation between the 2 given coefficients 
    '''
    dev = np.abs(coef1 - coef2)
    max_dev = np.max(dev)
    return max_dev



if __name__ == "__main__":


    ###################################################################################################################

    #PLOTTING THE FUNCTIONS

    x = np.linspace(-2*np.pi,4*np.pi,300)
    #fourier will handle only (0,2pi) and make that periodic so plot this also
    plotdata(x, exponential(x),xname= 'x', yname ='$exp^{x}$' , path = 'imgs/exp_plot',\
        label ='True function' ,plottype='semilogy',icon='g-', clear= False)

    xnew =np.linspace(0,2*np.pi,100)
    ynew = exponential(xnew)
    #note its 3 peroids so 
    y = np.tile(ynew, 3)
    plotdata(x,y,xname ='x', yname='log (exp x)', path = 'imgs/exp_plot',\
        label= 'periodic fucntion', plottype='semilogy', icon='r-')

    plotdata(x, coscos(x), xname ='x', yname='cos(cos(x))', path = 'imgs/cos_plot',\
        label= 'cos(cos(x)', plottype='plot', icon='r-')

    #############################################################################################################

    #PLOTTING FOURIER COEFFICIENTS 

    Fcoef_cos = fourier_coeff(51,coscos)
    Fcoef_exp = fourier_coeff(51,exponential)

    plotdata(x = range(1,52), y1 = np.abs(Fcoef_cos), xname = 'coeff', yname = 'log(value)', path = 'imgs/coef_cos_semilog',plottype='semilogy')
    plotdata(x = range(1,52), y1 = np.abs(Fcoef_exp), xname = 'coeff', yname = 'log(value)', path = 'imgs/coef_exp_semilog',plottype='semilogy')

    plotdata(x = range(1,52), y1 = np.abs(Fcoef_cos), xname = 'log(coeff)', yname = 'log(value)', path = 'imgs/coef_cos_log',plottype='loglog')
    plotdata(x = range(1,52), y1 = np.abs(Fcoef_exp), xname = 'log(coeff)', yname = 'log(value)', path = 'imgs/coef_exp_log',plottype='loglog')
    
    ##############################################################################################################

    #PLOTTING FOURIER COEFFICIENTS USING LEAST SQAURE APPROACH

    Aexp, bexp = leastSquareCoef(exponential)
    Lcoef_exp = lstsq(Aexp,bexp)[0]

    Acos, bcos = leastSquareCoef(coscos)
    Lcoef_cos = lstsq(Acos,bcos)[0]

    plotdata(x = range(1,52), y1 = np.abs(Lcoef_cos),y2 = np.abs(Fcoef_cos),icon=['go','ro'], xname = 'coeff', yname = 'log(value)',\
         label= ['lstq values', 'integration'], path = 'imgs/coef_cos_semilog2',plottype='semilogy')

    plotdata(x = range(1,52), y1 = np.abs(Lcoef_exp),y2 = np.abs(Fcoef_exp),icon=['go','ro'], xname = 'coeff', yname = 'log(value)',\
         label= ['lstq values', 'integration'], path = 'imgs/coef_exp_semilog2',plottype='semilogy')

    plotdata(x = range(1,52), y1 = np.abs(Lcoef_cos),y2 =np.abs(Fcoef_cos), icon=['go','ro'], xname = 'coeff', yname = 'log(value)',\
         label= ['lstq values', 'integration'], path = 'imgs/coef_cos_log2',plottype='loglog')

    plotdata(x = range(1,52), y1 = np.abs(Lcoef_exp),y2 = np.abs(Fcoef_exp),icon=['go','ro'], xname = 'coeff', yname = 'log(value)',\
         label= ['lstq values', 'integration'], path = 'imgs/coef_exp_log2',plottype='loglog')
    ###########################################################################################################################

    #OBTAINING PREDICTED VALUES

    predicted_cos =  np.matmul(Acos,Lcoef_cos) 

    predicted_exp  = np.matmul(Aexp, Lcoef_exp)

    xcoord = np.linspace(0,2*np.pi, 401)[:-1]

    plt.plot(xcoord, predicted_cos,'go',ms =4, label = 'predicted', )
    plt.plot(xcoord, coscos(xcoord), label = 'True')
    plt.grid(True)
    plt.xlabel('x(linear)')
    plt.ylabel('y(linear)')
    plt.legend(loc = 'upper right')
    plt.savefig('imgs/Figure1.png',bbox_inches='tight')
    plt.clf()

    plt.semilogy(xcoord, predicted_exp,'go',ms =4, label = 'predicted', )
    ycoord = np.tile(exponential(xcoord), 3)
    plt.semilogy(xcoord,exponential(xcoord), label = 'True')
    plt.xlabel('x(linear)')
    plt.ylabel('y(log)')
    plt.grid(True)
    plt.legend(loc = 'upper right')
    plt.savefig('imgs/Figure0.png',bbox_inches='tight')
    plt.clf()

    #################################################################################################################################

    #COMPUTING ERROR IN THE TWO ESTIMATES

    print('Max deviation in the 2 methods for computing cos(cos(x)) coefficients is ',compare(Lcoef_cos, Fcoef_cos))
    print('Max deviation in the 2 methods for computing e^x coefficients is ',compare(Lcoef_exp, Fcoef_exp))


    ##################################################################################################################################













