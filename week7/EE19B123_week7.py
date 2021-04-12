'''
EE2703 assignment 7
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: April 10, 2021

'''
import numpy as np 
import matplotlib.pyplot as plt 
import argparse
import mpl_toolkits.mplot3d.axes3d as p3
import os
import random
import gc
from tqdm import tqdm
import scipy.signal as sp
os.chdir('/home/tanay/Documents/sem4/EE2703/week7')#TODO remove this befor submitting

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
            if kwargs.get('legend'):
                plt.legend(kwargs['label'], loc ='upper right')
            function(*args, **kwargs)
            plt.savefig(kwargs['path']+'.png',bbox_inches='tight')
            print('File saved at {}'.format(kwargs['path']))
            plt.clf()
        return fcall

class plot:
    def __init__(self):
        pass
    @save
    def plot(self, x, y,**kwargs):
        plt.plot(x,y)


class spring(plot):
    def __init__(self,coeff):
        '''
        @Params:
        coeff: (list) The coefficients of the differential eqn
        '''
        super(spring, self).__init__()
        self.coef  = coeff
        

    def H(self, freq , alpha):
        '''
        @Params:
        freq:  frequecy of the system
        alpha: decay constant
        '''
        p= np.polymul(self.coef,[1,-2*alpha,freq*freq + alpha*alpha])
        return sp.lti([1,-1*alpha],p)

    def LTI_system(self, FreqRange):
        '''
        @Params:
        FreqRange : list of possible frequencies : np.linspace(1.4,1.6,5)
        '''
        l = []
        for f in FreqRange:
            h = sp.lti([1],self.coef)
            time = np.linspace(0,50,5001)
            u = np.cos(f*time)*np.exp(-0.05*time)
            t,x,_ = sp.lsim(h,u,time)
            self.plot(t, x)
            l.append("freq =" + str(f))

    @staticmethod
    def system(num, den):
        '''
        @Params:
        num : numerator coefficients 
        den : denominator coefficients
        '''
        X = sp.lti(num,den)
        t,x = sp.impulse(X,None,np.linspace(0,50,5001))
        return t, x

    @staticmethod
    def v(t):
        return np.cos(1000*t) -np.cos(1e6*t)

    def TwoPortNetwork(self,time,R, L,C,vi = v):
        H = sp.lti([1],[L*C,R*C,1])
        w,S,phi = H.bode()
        fig,(ax1,ax2) = plt.subplots(2,1)
        ax1.set_title("Magnitude response")
        ax1.semilogx(w,S)
        ax2.set_title("Phase response")
        ax2.semilogx(w,phi)
        plt.show()
        return sp.lsim(H,vi(time),time)

if __name__=='__main__':
    #Q1 
    laplace_solver = spring([1.0,0,2.25])
    h = laplace_solver.H(1.5, -0.5)
    t,x = sp.impulse(h,None,np.linspace(0,50,5001))
    #plot

    #Q2
    laplace_solver = spring([1.0,0,2.25])
    h = laplace_solver.H(1.5, -0.05)
    t,x = sp.impulse(h,None,np.linspace(0,50,5001))
    #plot

    #Q3


    #Q4
    t,x  = laplace_solver.system([1,0,2],[1,0,3,0])
    t,y = laplace_solver.system([2],[1,0,3,0])
    #plot both


    #Q5
    t=np.linspace(0,30e-6,10000)
    t,y,_ = laplace_solver.TwoPortNetwork(t,100, 1e-6 , 1e-6)






