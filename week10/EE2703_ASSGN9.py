##**********************************************************************************************************************************************************************
#                                                     EE2703 Applied Programming Lab 2021
#                                                                 Assignment 8
#
# Purpose  : To find discrete fourier transforms of functions which are discontinuous when periodically extended using windowing.
# Author   : Neham Jain (EE19B084)
# Input    : No command line input is required
# Output   : The graphs are saved in a directory called plots

##**********************************************************************************************************************************************************************

#Libraries used in our program
from pylab import *
import matplotlib.pyplot as plt 
import os
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm


#Setting Default Parameters for the Plots
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = 12,8


#Helper Class for Plotting the Fourier Transforms
class General_Plotter():
    ''' Class used for plotting different plots. Shortens the code by quite a bit'''
    
    fig_num=0   #Defined static variable for the figure number
    def __init__(self,xlabel1,ylabel1,ylabel2=None,title1=None,save_name=None):
        ''' xlabel,ylabel,title are used in every graph''' 

        self.xlabel1 = xlabel1
        self.ylabel1 = ylabel1
        
        self.xlabel2 = xlabel1
        self.ylabel2 = ylabel2
        
        self.title1=title1

        self.save_name=save_name
        self.fig=plt.figure(self.__class__.fig_num)
        self.__class__.fig_num+=1

    def general_funcs1(self,ax,xlim,ylim):
        ''' General functions for every graph'''
        
        ax[0].set_ylabel(self.ylabel1)
        ax[0].set_xlabel(self.xlabel1)
        ax[0].set_title(self.title1)
        ax[1].set_ylabel(self.ylabel2)
        ax[1].set_xlabel(self.xlabel2)
        if xlim is not None:
            ax[0].set_xlim(-xlim,xlim)
            ax[1].set_xlim(-xlim,xlim)
        if ylim is not None:
            ax[1].set_ylim(-ylim,ylim)

        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        self.fig.savefig("plots/"+self.save_name+".png")
        close()
    
    def general_funcs2(self,ax):
        ''' General functions for every graph'''
        
        ax.set_ylabel(self.ylabel1)
        ax.set_xlabel(self.xlabel1)
        ax.set_title(self.title1)
        self.fig.savefig("plots/"+self.save_name+".png")
        close()
    
    def general_plot(self,x1,y1,mark):
        axes=self.fig.add_subplot(111)
        axes.plot(x1,y1,mark)
        self.general_funcs2(axes)

    def semilog(self,x,y,xlim1,xlim2,ylim1,ylim2):
        
        axes=self.fig.add_subplot(111)
        axes.semilogx(x,y)
        axes.set_xlim(xlim1,xlim2)
        axes.set_ylim(ylim1,ylim2)
        self.general_funcs2(axes)

    def plot_fft(self,w,mag,phi,xlim=None,ylim=None):
        ''' Helper Function for plotting the fft of a given signal'''
        
        axes=self.fig.subplots(2,1)
        axes[0].plot(w,mag,lw=2)
        axes[1].plot(w,phi,'ro')
        self.general_funcs1(axes,xlim,ylim)

def surface_plotter(Y_mags,Y_angles,savename):
    #Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    t=np.linspace(-np.pi,np.pi,1025);t=t[:-1]
    fmax = 1/(t[1]-t[0])
    t=t[::64]
    w=np.linspace(-fmax*np.pi,fmax*np.pi,64+1);w=w[:-1]
    t,w=np.meshgrid(t,w)
    
    surf=ax.plot_surface(w,t,Y_mags.T,cmap=plt.get_cmap('jet'),linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.ylabel("Frequency")
    plt.xlabel("time")
    plt.savefig("plots/"+savename+"mag.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf=ax.plot_surface(w,t,Y_angles.T,cmap=plt.get_cmap('jet'),linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.ylabel("Frequency")
    plt.xlabel("time")
    plt.savefig("plots/"+savename+"angles.png")
    close()

#helper functions
def calculate_fft(lim,n,f,t_=0,t_lims = False,windowing= False,xlim1=10,title1=r"$\sin\left(\sqrt{2}t\right)$",xlabel1 = r"$\omega$",ylabel1= r"$|Y|$", ylabel2 = r"Phase of $Y$",savename = "abc.png",semilog=False,plot_flag=True):
    
    if(t_lims):
        t = t_
    else:
        t=linspace(-lim,lim,n+1)[:-1]
    dt=t[1]-t[0]
    fmax=1/dt
    y = f(t)
    if (windowing):
        m=arange(n)
        wnd=fftshift(0.54+0.46*cos(2*pi*m/n))
        y = y*wnd

    y[0]=0 # the sample corresponding to -tmax should be set zeroo
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/float(n)
    w=linspace(-pi*fmax,pi*fmax,n+1)[:-1]
    
    mag = abs(Y)
    ph = angle(Y)
    
    if semilog:
        p2=General_Plotter(xlabel1,ylabel1,ylabel2,title1,savename)
        p2.semilog(abs(w),20*log10(abs(Y)),1,10,-20,0)

    else:
        if plot_flag:
            p1=General_Plotter(xlabel1,ylabel1,ylabel2,title1,savename)
            p1.plot_fft(w,mag,ph,xlim1)

    return w,Y



#defining cos^3
def cos3(t,w0=0.86):
    return (cos(w0*t))**3

#Defining Normal Cosine
def cosine(t,w0=1.5,delta=0.5):
    return cos(w0*t + delta)

#Cosine function with noise added to it
def noisycosine(t,w0=1.5,delta=0.5):
    return cos(w0*t + delta) + 0.1*np.random.randn(128)

def sin_sqrt(t):
    return sin(sqrt(2)*t)

def chirp(t):
    return cos(16*(1.5 + t/(2*pi))*t) 

def identical(t):
    return t

def est_omega(w,Y):
    ii = where(w>0)
    omega = (sum(abs(Y[ii])**2*w[ii])/sum(abs(Y[ii])**2))#weighted average
    print ("omega = ", omega)

def est_delta(w,Y,sup = 1e-4,window = 1):
    ii_1=np.where(np.logical_and(np.abs(Y)>sup, w>0))[0]
    np.sort(ii_1)
    points=ii_1[1:window+1]
    print (np.sum(np.angle(Y[points]))/len(points))#weighted average for first 2 points



def main():

    #Assignment Examples 
    
    #Example 1
    calculate_fft(pi,64,sin_sqrt,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$", ylabel2 = r"Phase of $Y$",savename = "sin(sqrt(2)",title1=r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
    
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
    calculate_fft(pi,64,identical,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "ramp",title1=r"Spectrum of a digital ramp",semilog=True)

    #Example 5
    n=arange(64)
    wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
    y=sin(sqrt(2)*t1)*wnd
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
    calculate_fft(pi,64,sin_sqrt,xlim1=8,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "sin_sqrt2_window1",title1=r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$",windowing=True)

    #Example 7
    calculate_fft(4*pi,256,sin_sqrt,xlim1=4,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "sin_sqrt2_window2",title1=r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$",windowing=True)

    #Assignment Questions 

    #cos^3(t)

    #Without Windowing 
    calculate_fft(4*pi,256,cos3,xlim1=3,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "cos^3_wo",title1= r"Spectrum of $cos^3(w_0t)$",windowing=False)
    #With Windowing
    calculate_fft(4*pi,256,cos3,xlim1=3,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "cos^3_with",title1= r"Spectrum of $cos^3(w_0t)$",windowing=True)

    #Question 3
    
    w,Y = calculate_fft(pi,128,cosine,xlim1=3,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "normal_cosine",title1= r"Spectrum of $cos(w_0t + \delta)$",windowing=True)
    est_omega(w,Y)
    est_delta(w,Y)

    #Question 4
    
    w,Y = calculate_fft(pi,128,noisycosine,xlim1=3,xlabel1 = r"$\omega$",ylabel1= r"$|Y|$ (dB)", ylabel2 = r"Phase of $Y$",savename = "noisy_cosine",title1= r"Spectrum of $cos(w_0t + \delta)$ with noise added",windowing=True)
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










#if file is run directly
if __name__=="__main__":
    
    #Creating directory for storing plots
    os.makedirs("plots",exist_ok=True)
    #Running the main function
    print("Getting Results...")
    main()
