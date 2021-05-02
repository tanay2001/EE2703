'''
EE2703 assignment 9
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: May 1, 2021
'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from numpy.fft import fftshift, fft
os.chdir('/home/tanay/Documents/sem4/EE2703/week9')#TODO remove this befor submitting
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
    plt.ylabel(r"$|y|$", size = 12)
    plt.title(f"Spectrum of {kwargs['func']}")
    plt.grid(True)
    plt.subplot(2,1,2)
    ii = np.where(np.abs(y)>kwargs['phase_limit'])
    plt.plot(x[ii], np.angle(y[ii]), 'ro', lw=2)
    plt.xlim([-kwargs['xlim'],kwargs['xlim']])
    plt.ylabel(r"Phase of $Y$", size=12)
    plt.xlabel(r"$k$", size=12)
    plt.grid(True)

if __name__ =='__main__':
    #Examples 1 ========================================
    x=np.linspace(0,2*np.pi,128)
    y=np.sin(5*x)
    Y=np.fft.fft(y)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(abs(Y),lw=2)
    plt.grid(True)
    plt.ylabel(r"$|Y|$")
    plt.title(r"Spectrum of $\sin(5t)$")
    plt.subplot(2,1,2)
    plt.plot(np.unwrap(np.angle(Y)),lw=2)
    plt.ylabel("Phase of Y")
    plt.xlabel("k")
    plt.grid(True)

    #Examples 2 ========================================
    x=np.linspace(0,2*np.pi,129)[:-1]
    y=np.sin(5*x)
    Y=np.fft.fftshift(np.fft.fft(y))/128.0
    w=np.linspace(-64,63,128)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w,abs(Y),lw=2)
    plt.xlim([-10,10])
    plt.ylabel(r"$|Y|$")
    plt.title(r"Spectrum of $\sin(5t)$")
    plt.grid(True)
    plt.subplot(2,1,2)
    ii=np.where(np.abs(Y)>1e-3)
    plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
    plt.xlim([-10,10])
    plt.ylabel(r"Phase of $Y$",size=16)
    plt.xlabel(r"$k$",size=16)
    plt.grid(True)

    #Examples 3 ===========================================
    t=np.linspace(0,2*np.pi,129);t=t[:-1]
    y=(1+0.1*np.cos(t))*np.cos(10*t)
    Y=np.fft.fftshift(np.fft.fft(y))/128.0
    w=np.linspace(-64,63,128)

    # Figure 1
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w,abs(Y),lw=2)
    plt.xlim([-15,15])
    plt.ylabel(r"$|Y|$",size=16)
    plt.title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
    plt.grid(True)
    
    # Figure 2
    plt.subplot(2,1,2)
    #plt.plot(w,np.angle(Y),'ro',lw=2)
    ii=np.where(np.abs(Y)>1e-3)
    plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
    plt.xlim([-15,15])
    plt.ylim([-3,3])
    plt.ylabel(r"Phase of $Y$",size=16)
    plt.xlabel(r"$\omega$",size=16)
    plt.grid(True)

    #Assignment Questions 
    def cos3(x):
        return np.cos(x)**3

    def sin3(x):
        return np.sin(x)**3

    def gauss(x):
        return  np.exp(-x**2/2)
    
    def coscos(x):
        return np.cos(20*x+5*np.cos(x))

    func_dict = {'cos^3' : cos3,
                'sin^3' : sin3,
                'coscos' : coscos,
                'gauss' : gauss }

    def dft(func,N=512,steps = 513, r=4*np.pi, phase_limit=1e-3, xlim=40, w_lim=64):
        t = np.linspace(-r,r,steps)[:-1]
        y= func_dict[func](t)
        Y = fftshift(fft(y))/N
        w = np.linspace(-w_lim,w_lim,steps)[:-1]
        if func == 'gauss' : 
            Y = fftshift(np.abs(fft(y)))/N
            # Normalizing 
            Y = Y*np.sqrt(2*np.pi)/np.max(Y)
            Y_ = np.exp(-w**2/2)*np.sqrt(2*np.pi)
            print(f"max error is {np.abs(Y-Y_).max()}")

        customplot(w,Y,
        func =func,
        path ='imgs/Q'+func,
        xlim = xlim ,
        phase_limit = phase_limit)
        return Y,w

    Y,w = dft('cos^3',xlim=15, steps =129, N = 128, w_lim=16)
    Y,w = dft('sin^3',xlim=15)
    Y,w = dft('coscos',xlim=40)
    Y1,w =dft('gauss',N=512,r=8*np.pi,w_lim=32,xlim=10)

