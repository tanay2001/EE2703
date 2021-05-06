'''
EE2703 assignment 9
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: May 1, 2021
'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from pylab import *
from numpy.fft import fftshift, fft, ifft
#os.chdir('/home/tanay/Documents/sem4/EE2703/week9')#TODO remove this befor submitting
#===================================================
#creating an imgs directory to store all png files
if os.path.isdir('imgs'):
    pass
else:
    os.mkdir('imgs')
#===================================================
def save(function):
        '''
        decorator function
        adds plot label/title/legend/save file
        '''
        def fcall(*args, **kwargs): 
            function(*args, **kwargs)
            if kwargs.get('legend'):
                plt.legend(loc ='upper right')
            plt.savefig(kwargs['path']+'.png',bbox_inches='tight')
            print('File saved at {}'.format(kwargs['path']))
            plt.clf()
        return fcall
@save
def customplot(x,y,**kwargs):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x,np.abs(y),lw=2)
    plt.xlim([-kwargs['xlim'],kwargs['xlim']])
    plt.xlabel('Frequency in rad/s')
    plt.ylabel(r"$|y|$", size = 12)
    plt.title(f"Spectrum of {kwargs['func']}")
    plt.grid(True)
    plt.subplot(2,1,2)
    ii = np.where(np.abs(y)>kwargs['phase_limit'])
    plt.plot(x[ii], np.angle(y[ii]), 'ro', lw=2)
    plt.xlim([-kwargs['xlim'],kwargs['xlim']])
    if kwargs.get('ylimit'):
        plt.ylim([-kwargs['ylimit'],kwargs['ylimit']])
    plt.ylabel(r"Phase of $Y$", size=12)
    plt.xlabel('Frequency in rad/s', size = 12)
    plt.grid(True)

if __name__ =='__main__':
    #Example 0 ==========================================================
    x=np.random.rand(100)
    X=fft(x)
    y=ifft(X)
    c_[x,y]
    maxError = max(np.abs(y-x))
    print('Magnitude of maximum error between actual and computed values of the random sequence:', maxError)


    #Assignment Questions ==================================================
    def cos3(x):
        return np.cos(x)**3

    def sin3(x):
        return np.sin(x)**3

    def gauss(x):
        return  np.exp(-x**2/2)
    
    def coscos(x):
        return np.cos(20*x+5*np.cos(x))

    def sin5(x):
        return np.sin(5*x)

    def modulated(x):
        return (1+0.1*np.cos(x))*np.cos(10*x)

    func_dict = {'sin':sin5,
                'modul':modulated,
                'cos^3' : cos3,
                'sin^3' : sin3,
                'coscos' : coscos,
                'gauss' : gauss }

    def dft(func,N=512,steps = 513, r=4*np.pi, phase_limit=1e-3, xlim=40, w_lim=64, ylimit=None):
        t = np.linspace(-r,r,steps)[:-1]
        y= func_dict[func](t)
        Y = fftshift(fft(y))/N
        w = np.linspace(-w_lim,w_lim,steps)[:-1]
        if func == 'gauss' : 
            Y = fftshift(np.abs(fft(y)))/N
            # Normalizing 
            Y = Y*np.sqrt(2*np.pi)/np.max(Y)
            Y_ = np.exp(-w**2/2)*np.sqrt(2*np.pi)
            print(f"max error is {max(np.abs(Y-Y_))}")

        customplot(w,Y,
        func =func,
        path ='imgs/Q'+func,
        xlim = xlim ,
        ylimit = ylimit,
        phase_limit = phase_limit)
        return Y,w

    #sample Questions =========================================
    Y1,w1= dft('sin', xlim=10)
    Y2, w2 = dft('modul',xlim=40, ylimit = 1)
    #assigments questions =====================================
    Y3,w3 = dft('cos^3',xlim=15, steps= 129 , w_lim=16, N = 128, ylimit=1)
    Y4,w4 = dft('sin^3',xlim=15)
    Y5,w5 = dft('coscos',xlim=40)
    Y6,w6 =dft('gauss',N=512,r=8*np.pi,w_lim=32,xlim=10)

