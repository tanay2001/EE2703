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
#os.chdir('/home/tanay/Documents/sem4/EE2703/week10')#TODO remove this befor submitting
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
            if kwargs.get('path'):
                plt.savefig(kwargs['path']+'.png',bbox_inches='tight')
                print('File saved at {}'.format(kwargs['path']))
            plt.clf()
        return fcall

@saveplot
def simplePlot(x,y,**kwargs):
    plt.plot(x,y)

@saveplot
def semilogPlot(x,y,**kwargs):
    plt.semilogx(x,y)
    plt.xlim(kwargs['xlim1'], kwargs['xlim2'])
    plt.ylim(kwargs['ylim1'], kwargs['ylim2'])

@saveplot
def plotter(w,mag,phi,**kwargs):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w,mag,lw=2)
    plt.grid(True)
    if kwargs.get('xlim'):
        plt.xlim(-kwargs['xlim'],kwargs['xlim'])
    plt.ylabel(kwargs['ylabel1'])
    plt.subplot(2,1,2)
    plt.plot(w,phi,'ro')
    plt.grid(True)
    if kwargs.get('xlim'):
        plt.xlim(-kwargs['xlim'],kwargs['xlim'])
    plt.xlabel(kwargs['xlabel1'])
    plt.ylabel(kwargs['ylabel2'])
    kwargs['xlabel'] =None
    kwargs['ylabel']= None

def surfaceplots(y_mags,y_phase,path):
    '''
    3D plotter fucntions for mag , phase
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    t=np.linspace(-np.pi,np.pi,1025)[:-1]
    fmax = 1/(t[1]-t[0])
    t=t[::64]
    w=np.linspace(-fmax*np.pi,fmax*np.pi,64+1)[:-1]
    t,w=np.meshgrid(t,w)
    
    surf=ax.plot_surface(w,t,y_mags.T,cmap=plt.get_cmap('jet'),linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.savefig("imgs/"+path+"mag.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf=ax.plot_surface(w,t,y_phase.T,cmap=plt.get_cmap('jet'),linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.savefig("imgs/"+path+"ang.png")
    plt.clf()
    close()

# ==============================================================================================

def FFTCalci(f,N,plotting = True, **kwargs):

    if  not kwargs.get('use_new_t'):
        t=np.linspace(-kwargs['tlim'],kwargs['tlim'],N+1)[:-1]
    else:
        t = kwargs['time']

    dt=t[1]-t[0]
    fMax=1/dt
    y = f(t)
    if kwargs.get('windowing'):
        m=np.arange(N)
        window=fftshift(0.54+0.46*np.cos(2*np.pi*m/N))
        y = y*window

    y[0]=0
    y=fftshift(y) 
    Y=fftshift(fft(y))/float(N)
    w=np.linspace(-np.pi*fMax,np.pi*fMax,N+1)[:-1]
    
    magnitude = np.abs(Y)
    phase = np.angle(Y)
    if plotting:
        plotter(w,magnitude,phase, **kwargs)

    return w,Y


# defining all functions that will be needed

def cosine(t,w0=1.5,delta=0.5):
    return np.cos(w0*t + delta)

def NoisyCos(t,w0=1.5,delta=0.5):
    return np.cos(w0*t + delta) + 0.1*np.random.randn(len(t))

sin_sqrt = lambda t : np.sin(np.sqrt(2)*t)
cos3 = lambda x: np.cos(0.86*x)**3
chirped = lambda t: np.cos(16*(1.5 + t/(2*np.pi))*t) 


# fucntions used to estimate omega and delta using method given in assignment
def estimate_omega(w,Y):
    ids = np.where(w>0)
    omega = (sum(np.abs(Y[ids])**2*w[ids])/sum(np.abs(Y[ids])**2))
    print("Estimated omega is", omega)

def estimate_delta(w,Y):
    ids=np.where(np.logical_and(np.abs(Y)>1e-4, w>0))[0]
    np.sort(ids)
    points=ids[1:2]
    print(np.sum(np.angle(Y[points]))/len(points))


if __name__=='__main__':

    #Given Examples=========================================
    #Ex1
    FFTCalci(sin_sqrt, 64,
            tlim = np.pi,
            xlabel1 = r"$\omega$",
            ylabel1= r"$|Y|$", 
            ylabel2 = r"Phase of $Y$",
            path = "imgs/sin_sqrt",
            xlim = 10,
            title1=r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
    
    #Ex2
    t1=np.linspace(-np.pi,np.pi,65)[:-1]
    t2=np.linspace(-3*np.pi,-np.pi,65)[:-1]
    t3=np.linspace(np.pi,3*np.pi,65)[:-1]
    y=np.sin(np.sqrt(2)*t1)

    plot(t1,np.sin(np.sqrt(2)*t1),"b")
    plot(t2,np.sin(np.sqrt(2)*t2),"r")
    plot(t3,np.sin(np.sqrt(2)*t3),"r")
    ylabel(r"$y$")
    xlabel(r"$t$")
    title(r"$\sin\left(\sqrt{2}t\right)$")
    savefig("imgs/sin_sqrt_plot.png")
    close()

    #Ex3
    plot(t1,y,'bo',lw=2)
    plot(t2,y,'ro',lw=2)
    plot(t3,y,'ro',lw=2)
    ylabel(r"$y$")
    xlabel(r"$t$")
    title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
    savefig("imgs/wrapped_sin.png")
    close()

    #Ex4
    w,Y = FFTCalci(lambda x: x, 64,
    tlim = np.pi,plotting = False)
        
    semilogPlot(abs(w),20*np.log10(abs(Y)),
        xlim1 = 1,
        xlim2 = 10,
        ylim1 = -20,
        ylim2 = 0,
        xlabel1 = r"$\omega$",
        ylabel1= r"$|Y|$ (dB)", 
        ylabel2 = r"Phase of $Y$",
        path = "imgs/ramp",
        title1=r"Spectrum of a digital ramp",
        semilog=True)


    #Ex5
    n=np.arange(64)
    wnd=fftshift(0.54+0.46*np.cos(2*np.pi*n/63))
    y=np.sin(np.sqrt(2)*t1)*wnd
    plot(t1,y,'bo',lw=2)
    plot(t2,y,'ro',lw=2)
    plot(t3,y,'ro',lw=2)
    ylabel(r"$y$",size=16)
    xlabel(r"$t$",size=16)
    title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
    grid(True)
    savefig("imgs/windowed_sin.png")
    close()

    #Ex6
    FFTCalci(sin_sqrt,64,
            tlim = np.pi,
            xlim=8,
            xlabel1 = r"$\omega$",
            ylabel1= r"$|Y|$ (dB)", 
            ylabel2 = r"Phase of $Y$",
            path = "imgs/sin_window1",
            title1=r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$",
            windowing=True)

    #Ex7
    FFTCalci(sin_sqrt,256,
            tlim = 4*np.pi,
            xlim=4,
            xlabel1 = r"$\omega$",
            ylabel1= r"$|Y|$ (dB)", 
            ylabel2 = r"Phase of $Y$",
            path = "imgs/sin_window2",
            title1=r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$",
            windowing=True)

    #================================================= Assignment Questions ============================================================================

    # Q2============================================================

    #With Hamming Window
    FFTCalci(cos3, 256,
                tlim = 4*np.pi,
                xlim1=3,
                xlabel1 = r"$\omega$",
                ylabel1= r"$|Y|$ (dB)", 
                ylabel2 = r"Phase of $Y$",
                path = "imgs/cos^3_with",
                xlim = 3,
                title1= r"Spectrum of $cos^3(w_0t)$",
                windowing=True)

    #Without Hamming Window
    FFTCalci(cos3, 256,
                    tlim = 4*np.pi,
                    xlim1=3,
                    xlabel1 = r"$\omega$",
                    ylabel1= r"$|Y|$ (dB)", 
                    ylabel2 = r"Phase of $Y$",
                    path = "imgs/cos^3_wo",
                    xlim =3,
                    title1= r"Spectrum of $cos^3(w_0t)$",
                    windowing=False)

    #Q3 =================================================================
    
    w,Y = FFTCalci(cosine,128,
                        tlim = np.pi,
                        xlim1=3,
                        xlabel1 = r"$\omega$",
                        ylabel1= r"$|Y|$ (dB)", 
                        ylabel2 = r"Phase of $Y$",
                        path = "imgs/normal_cosine",
                        xlim = 3,
                        title1= r"Spectrum of $cos(w_0t + \delta)$",
                        windowing=True)
    estimate_omega(w,Y)
    estimate_delta(w,Y)

    #Q4 ========================================================================
    
    w,Y = FFTCalci(NoisyCos,128,
                    tlim = np.pi,
                    xlim1=3,
                    xlabel1 = r"$\omega$",
                    ylabel1= r"$|Y|$ (dB)", 
                    ylabel2 = r"Phase of $Y$",
                    path = "imgs/noisy_cosine",
                    xlim = 3,
                    title1= r"Spectrum of $cos(w_0t + \delta)$ with noise added",
                    windowing=True)
    estimate_omega(w,Y)
    estimate_delta(w,Y)

    #Q5 ========================================================================
    FFTCalci(chirped,1024,
            tlim = np.pi,
            xlim1=60,
            xlabel1 = r"$\omega$",
            ylabel1= r"$|Y|$ (dB)", 
            ylabel2 = r"Phase of $Y$",
            path = "imgs/chirp_with",
            xlim = 60,
            title1= r"Spectrum of chirp function",
            windowing=True)
    FFTCalci(chirped,1024,
            tlim = np.pi,
            xlim1=60,
            xlabel1 = r"$\omega$",
            ylabel1= r"$|Y|$ (dB)", 
            ylabel2 = r"Phase of $Y$",
            path = "imgs/chirp_wo",
            xlim = 60,
            title1= r"Spectrum of chirp function",
            windowing=False)

    #Q6 =============================================================

    t=linspace(-np.pi,np.pi,1025)[:-1]
    tarr=np.split(t,16)

    Ymag_win=np.zeros((16,64))
    Yangle_win=np.zeros((16,64))
    
    Ymag_no_win=np.zeros((16,64))
    Yangle_no_win=np.zeros((16,64))
     
    #Splitting array and doing fft
    for i in range(len(tarr)):

        w,Y =  FFTCalci(chirped,64,
                    time = tarr[i],
                    use_new_t = True,
                    xlabel1 = r"$\omega$",
                    ylabel1= r"$|Y|$", 
                    ylabel2 = r"Phase of $Y$",
                    path = None,
                    windowing=False)

        Ymag_no_win[i] =  abs(Y)
        Yangle_no_win[i] =  np.angle(Y)

        w,Y =  FFTCalci(chirped,64,
                        time = tarr[i],
                        use_new_t = True,
                        xlabel1 = r"$\omega$",
                        ylabel1= r"$|Y|$", 
                        ylabel2 = r"Phase of $Y$",
                        path = None,
                        windowing=True)

        Ymag_win[i] =  abs(Y)
        Yangle_win[i] =  np.angle(Y)

    surfaceplots(Ymag_no_win,
                Yangle_no_win,
                path = "non_windowed_surface")
    surfaceplots(Ymag_win,
                Yangle_win,
                path = "windowed_surface")




