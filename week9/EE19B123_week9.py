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

        customplot(w,Y,
        func =func,
        path ='imgs/Q'+func,
        xlim = xlim ,
        ylimit = ylimit,
        phase_limit = phase_limit)

    def dft_gaussian(func,threshold=1e-6,N=128):
        '''
        As questions demainds accuracy till 6 digits we need to iteratively keeps computing till it error falls
        '''
        T = 8*np.pi
        Y_old = 0
        while True:
            #Time 
            dt = T/N
            #Frequency 
            dw = 2*np.pi/T
            W = N*dw
            t = np.linspace(-T/2,T/2,N+1)[:-1]
            w = np.linspace(-W/2,W/2,N+1)[:-1]
            y = gauss(t)
            Y_new = dt/(2*np.pi) * fftshift(fft(ifftshift(y)))
            error = np.sum(np.abs(Y_new[::2]) - Y_old)
            Y_old = Y_new
            if error < threshold:
                customplot(w,Y_new,
                func =func,
                path ='imgs/Q'+func,
                xlim = 5,
                phase_limit = 1e-3 )
                print(f"Error in Gaussian case is {error}")
                break

            T*=2
            N*=2

    #sample Questions =========================================
    dft('sin', xlim=10)
    dft('modul',xlim=40, ylimit = 1)
    #assigments questions =====================================
    dft('cos^3',xlim=15, steps= 129 , w_lim=16, N = 128, ylimit=1)
    dft('sin^3',xlim=15)
    dft('coscos',xlim=40)
    dft_gaussian('gauss')

