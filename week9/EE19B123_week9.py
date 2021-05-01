'''
EE2703 assignment 9
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: May 1, 2021

'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
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
            if kwargs.get('xlabel'):
                plt.xlabel(kwargs['xlabel'])
            if kwargs.get('ylabel'):
                plt.ylabel(kwargs['ylabel'])
            if kwargs.get('title'):
                plt.title(kwargs['title'])
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
    plt.plot(abs(y),lw=2)
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(x,np.unwrap(np.angle(y)),lw=2)
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
