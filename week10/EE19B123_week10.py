'''
EE2703 assignment 10
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: May 5, 2021
'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from pylab import*
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.fft import fftshift, fft, ifft
import mpl_toolkits.mplot3d.axes3d as p3
os.chdir('/home/tanay/Documents/sem4/EE2703/week10')#TODO remove this befor submitting
#===================================================
#creating an imgs directory to store all png files
if os.path.isdir('imgs'):
    pass
else:
    os.mkdir('imgs')
#===================================================
def saveplot(function):
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


def semilog(self,x,y,xlim1,xlim2,ylim1,ylim2):   
    axes=self.fig.add_subplot(111)
    axes.semilogx(x,y)



def FFTCalci(f,N, **kwargs):
    
    t=np.linspace(-kwargs['tlim'],kwargs['tlim'],N+1)[:-1]

    dt=t[1]-t[0]
    fMax=1/dt
    y = f(t)
    if kwargs['windowing']:
        m=np.arange(n)
        wnd=fftshift(0.54+0.46*np.cos(2*np.pi*m/n))
        y = y*wnd

    y[0]=0
    y=fftshift(y) 
    Y=fftshift(fft(y))/float(n)
    w=np.linspace(-np.pi*fMax,np.pi*fMax,N+1)[:-1]
    
    magnitude = np.abs(Y)
    phase = np.angle(Y)
    
    if kwargs['semilog']:
        p2=General_Plotter(xlabel1,ylabel1,ylabel2,title1,savename)
        p2.semilog(abs(w),20*np.log10(abs(Y)),1,10,-20,0)

    if kwargs['plot_flag']:
        p1=General_Plotter(xlabel1,ylabel1,ylabel2,title1,savename)
        p1.plot_fft(w,mag,ph,xlim1)

    return w,Y





def cosine(t,w0=1.5,delta=0.5):
    return cos(w0*t + delta)

def NoisyCos(t,w0=1.5,delta=0.5):
    return cos(w0*t + delta) + 0.1*np.random.randn(len(t))

sin_sqrt = lambda t : np.sin(np.sqrt(2)*t)

cos3 = lambda x: np.cos( 0.86*t)**3

chirped = lambda t: np.cos(16*(1.5 + t/(2*np.pi))*t) 


def estimate_omega(w,Y):
    ii = np.where(w>0)
    omega = (sum(abs(Y[ii])**2*w[ii])/sum(abs(Y[ii])**2))
    print ("omega = ", omega)

def estimate_delta(w,Y,sup = 1e-4,window = 1):
    ii_1=np.where(np.logical_and(np.abs(Y)>sup, w>0))[0]
    np.sort(ii_1)
    points=ii_1[1:window+1]
    print (np.sum(np.angle(Y[points]))/len(points))



if __name__=='__main__':

    #Assignment Examples 
    
    #Example 1
    FFTCalci(pi,64,sin_sqrt,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$", ylabel2 = r"Phase of $Y$",savename = "sin(sqrt(2)",title1=r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
    
    #Example 2
    t1=linspace(-pi,pi,65);t1=t1[:-1]
    t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
    t3=linspace(pi,3*pi,65);t3=t3[:-1]
    y=sin(sqrt(2)*t1)

    plot(t1,sin(sqrt(2)*t1),"b",lw=2)
    plot(t2,sin(sqrt(2)*t2),"r",lw=2)
    plot(t3,sin(sqrt(2)*t3),"r",lw=2)
    ylabel(r"$y$")
    xlabel(r"$t$")
    title(r"$\sin\left(\sqrt{2}t\right)$")
    savefig("plots/sin(sqrt(2)_plot.png")
    close()

    #Example 3
    plot(t1,y,'bo',lw=2)
    plot(t2,y,'ro',lw=2)
    plot(t3,y,'ro',lw=2)
    ylabel(r"$y$")
    xlabel(r"$t$")
    title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
    savefig("plots/wrapped_sin.png")
    close()

    #Example 4
    FFTCalci(np.pi,64,identical,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "ramp",title1=r"Spectrum of a digital ramp",semilog=True)

    #Example 5
    n=np.arange(64)
    wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
    y=np.sin(sqrt(2)*t1)*wnd
    plot(t1,y,'bo',lw=2)
    plot(t2,y,'ro',lw=2)
    plot(t3,y,'ro',lw=2)
    ylabel(r"$y$",size=16)
    xlabel(r"$t$",size=16)
    title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
    grid(True)
    savefig("plots/windowed_sin.png")
    close()

    #Example 6
    FFTCalci(pi,64,sin_sqrt,xlim1=8,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "sin_sqrt2_window1",title1=r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$",windowing=True)

    #Example 7
    FFTCalci(4*np.pi,256,sin_sqrt,xlim1=4,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "sin_sqrt2_window2",title1=r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$",windowing=True)

    #================================================= Assignment Questions ============================================================================

    # Q2============================================================

    #With Hamming Window
    FFTCalci(4*np.pi,256,cos3,
                xlim1=3,
                xlabel1 = r"$\omega$",
                ylabel1= r"$|Y|$ (dB)", 
                ylabel2 = r"Phase of $Y$",
                savename = "cos^3_with",
                title1= r"Spectrum of $cos^3(w_0t)$",
                windowing=True)

    #Without Hamming Window
    FFTCalci(4*np.pi,256,cos3,
                    xlim1=3,
                    xlabel1 = r"$\omega$",
                    ylabel1= r"$|Y|$ (dB)", 
                    ylabel2 = r"Phase of $Y$",
                    savename = "cos^3_wo",
                    title1= r"Spectrum of $cos^3(w_0t)$",
                    windowing=False)

    #Q3 =================================================================
    
    w,Y = FFTCalci(np.pi,128,cosine,
                        xlim1=3,
                        xlabel1 = r"$\omega$",
                        ylabel1= r"$|Y|$ (dB)", 
                        ylabel2 = r"Phase of $Y$",
                        savename = "normal_cosine",
                        title1= r"Spectrum of $cos(w_0t + \delta)$",
                        windowing=True)
    estimate_omega(w,Y)
    estimate_delta(w,Y)

    #Q4 ========================================================================
    
    w,Y = calculate_fft(np.pi,128,NoisyCos,xlim1=3,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "noisy_cosine",title1= r"Spectrum of $cos(w_0t + \delta)$ with noise added",windowing=True)
    est_omega(w,Y)
    est_delta(w,Y)

    #Question 5
    calculate_fft(pi,1024,chirp,xlim1=60,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "chirp_with",title1= r"Spectrum of chirp function",windowing=True)
    calculate_fft(pi,1024,chirp,xlim1=60,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "chirp_wo",title1= r"Spectrum of chirp function",windowing=False)

    #Question 6

    t=linspace(-np.pi,np.pi,1025);t=t[:-1]
    t_arrays=split(t,16)
    
    Y_mags_non_windowed=np.zeros((16,64))
    Y_angles_non_windowed=np.zeros((16,64))
    
    Y_mags_windowed=np.zeros((16,64))
    Y_angles_windowed=np.zeros((16,64))
    

    #Splitting array and doing fft
    for i in range(len(t_arrays)):

        w,Y =  calculate_fft(lim = 10,t_ = t_arrays[i],t_lims=True,n = 64,f = chirp,windowing=False)
        Y_mags_non_windowed[i] =  abs(Y)
        Y_angles_non_windowed[i] =  angle(Y)

        w,Y =  calculate_fft(lim = 10,t_ = t_arrays[i],t_lims=True,n = 64,f = chirp,windowing=True)
        Y_mags_windowed[i] =  abs(Y)
        Y_angles_windowed[i] =  angle(Y)

    surface_plotter(Y_mags_non_windowed,Y_angles_non_windowed,"non_windowed_surface")
    surface_plotter(Y_mags_windowed,Y_angles_windowed,"windowed_surface")

    print("The plots are saved at: ",os.getcwd()+"/plots/")


