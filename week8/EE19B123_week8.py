'''
EE2703 assignment 8
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: April 17, 2021

'''
import numpy as np 
from sympy import *
import scipy.signal as sp 
init_session
import pylab
import matplotlib.pyplot as plt 
import os
os.chdir('/home/tanay/Documents/sem4/EE2703/week8')#TODO remove this befor submitting

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
def loglogplot(x,y, **kwargs):
    plt.loglog(x,y)
    plt.grid(True)

@save
def customplot(x,y,**kwargs):
    plt.plot(x,y)
    plt.grid(True)

@save
def plotmany(x,y,**kwargs):
    for id, i in enumerate(y):
        plt.plot(x,i, label = kwargs['labels'][id])
        plt.grid(True)

def sympy2LTI(x):
    X=simplify(x)
    n,d=fraction(X)
    n,d=Poly(n,s), Poly(d,s)
    numerator ,denominator=n.all_coeffs(), d.all_coeffs()
    num,den=[float(coeff) for coeff in numerator],[float(coeff) for coeff in denominator]
    H = sp.lti(num,den)
    return H

def lowpass(R1,R2,C1,C2,G,Vi):
    '''
    Function that implements the low pass filter
    '''
    s=symbols('s')
    A=Matrix([[0,0,1,-1/G],
            [-1/(1+s*R2*C2),1,0,0],
            [0,-G,G,1],
            [-1/R1-1/R2-s*C1,1/R2,0,s*C1]])

    b=Matrix([0,0,0,-Vi/R1])

    V = A.inv()*b
    return A,b,V

def highpass(R1,R3,C1,C2,G,Vi):
    '''
    Function that implements the High pass filter
    '''
    s=symbols('s')
    A=Matrix([[0,0,1,-1/G],
             [-(s*C2*R3)/(s*C2*R3+1),1,0,0],
             [0,-G,G,1],
             [-s*C1-s*C2-1/R1,s*C2,0,1/R1]])

    b=Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return A,b,V

def magnitude_response(v):
    w = np.logspace(0,8,801)
    ss = 1j*w
    hf = lambdify(s,v,'numpy')
    v_res = hf(ss)
    return v_res


if __name__ =='__main__':
    #========================================================================================
    s = symbols('s')
    # frequency range
    w = np.logspace(0,8,801)

    #Q0 Low pass filter implemntation 
    A,b,V_0 = lowpass(10000,10000,1e-9,1e-9,1.586,1) 
    Vo = V_0[3] #Output Vo

    v = magnitude_response(Vo)

    loglogplot(w, abs(v),
            title = 'Low pass Magnitude response',
            xlabel = 'omega',
            yalbel ='H(omega)',
            path = 'imgs/Q0')

    H = sympy2LTI(Vo)

    t = np.linspace(0,5e-3,10000)
    # ==============================================================
    #Q1
    t,y=sp.step(H,T=t)

    customplot(t,y,
        title = 'Step Response for low pass filter',
        xlabel = 't',
        ylabel = 'V_o(t)',
        path = 'imgs/Q1')

    #==================================================================
    #Q2
    vi = np.sin(2000*np.pi*t) + np.cos(2e6*np.pi*t)
    t,y,_ = sp.lsim(H,vi,t)
    customplot(t,y,
        title ='Low Pass Filter Response to given Input',
        xlabel ='t',
        ylabel ='V_o',
        path = 'imgs/Q2_lp')
    #===================================================================
    #Q3
    A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,1)
    Vo_HP = V[3] #Output Vo

    v_hp = magnitude_response(Vo_HP)
    loglogplot(w, abs(v_hp),
            title ='High Pass Magnitude Responce',
            ylabel ='H(omega)',
            xlabel ='omega',
            path = 'imgs/Q3a')

    H2=sympy2LTI(Vo_HP)

    t,y=sp.step(H2,T=t)
    customplot(t,y,
        title ='Step Response for High pass filter',
        xlabel = 't',
        ylabel = 'V_o',
        path = 'imgs/Q3b')
    #====================================================================
    #Q4 & Q5
    t_high=np.linspace(0,0.00001,10000)
    high_damp = lambda t : np.sin(2*np.pi*5e5*t)*np.exp(-0.5*t)*(t>0)
    customplot(t_high,high_damp(t_high),
        title = 'High frequency damped sinosoid',
        xlabel = 't',
        ylabel = 'Vi',
        path ='imgs/Q4a')

    t_low=np.linspace(0,10,1000)
    low_damp = lambda t : np.sin(2*np.pi*1*t)*np.exp(-0.5*t)*(t>0)
    customplot(t_low,low_damp(t_low),
        title = 'Low frequency damped sinosoid',
        xlabel = 't',
        ylabel = 'Vi',
        path ='imgs/Q4b')

    t_high,y_high,_ = sp.lsim(H2,high_damp(t_high),t_high)

    t_low,y_low,_ = sp.lsim(H2,low_damp(t_low),t_low)

    plotmany(t_high,[y_high, high_damp(t_high)],
            title = 'High Pass Filter Response to High Frequency Damped Signal',
            xlabel = 't',
            ylabel = 'Vi(t)',
            labels =['Vo','Vi'],
            legend = True,
            path = 'imgs/Q5a')
   
    plotmany(t_low,[y_low, low_damp(t_low)],
            title = 'High Pass Filter Response to Low Frequency Damped Signal',
            xlabel = 't',
            ylabel = 'Vi(t)',
            labels =['Vo','Vi'],
            legend = True,
            path = 'imgs/Q5b')



