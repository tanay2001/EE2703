'''
EE2703 assignment 5
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: March 17, 2021

'''

import numpy as np 
import matplotlib.pyplot as plt 
import argparse
import mpl_toolkits.mplot3d.axes3d as p3
import os
from pylab import contour, plot
import pylab
os.chdir('/home/tanay/Documents/sem4/EE2703/week5')

#########################
#creating an imgs directory to store all png files
if os.path.isdir('imgs'):
    pass
else:
    os.mkdir('imgs')
#########################

def add_stuf(function):
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
                plt.legend(('Calculated Error','Fit 1 (all iterations)','Fit 2 (>500 iterations)'))
            function(*args, **kwargs)
            if not kwargs.get('not_save'):
                plt.savefig(kwargs['path']+'.png',bbox_inches='tight')
                print('File saved at {}'.format(kwargs['path']))
                plt.clf()
        return fcall

class Plot:
    '''
    Class for plotting various types of graphs namely
    semilogy() 
    loglog()
    plot()
    plot3D()
    contour()
    '''
    def __init__(self):
        super(Plot, self).__init__()

    @add_stuf
    def semilogy(self,x,y, **kwargs):
        plt.semilogy(x,y,kwargs['marker'],ms = 4)

    @add_stuf
    def loglog(self,x,y, **kwargs):
        plt.loglog(x,y,kwargs['marker'],ms=4, label = kwargs['label'])

    @add_stuf
    def plot3D(self,x,y,z, **kwargs):
        fig = plt.figure() 
        ax=p3.Axes3D(fig) 
        ax.plot_surface(y, x, z.T, rstride=1, cstride=1, cmap=plt.cm.jet)
        ax.set_zlabel(kwargs['zlabel'])

    def plotMany(self, x, y,count, **kwargs):
        for i in range(count):
            if i <count-1:
                kwargs['not_save'] = True
            else:
                kwargs['not_save'] = False
            self.semilogy(x[i],y[i],**kwargs)

    @add_stuf
    def contour(self, X, Y,phi, **kwargs):
        ids = kwargs.get('ids')
        x = kwargs.get('x')
        y = kwargs.get('y')
        contour(Y,X[::-1],phi)
        plot(x[ids[0]],y[ids[1]],'ro')


    @add_stuf
    def quiver(self, X, Y,jx, jy, **kwargs):
        ids = kwargs.get('ids')
        x = kwargs.get('x')
        y = kwargs.get('y')
        fig,ax = plt.subplots()
        fig = ax.quiver(Y,X[::-1],jx,jy, scale = 5)
        plot(x[ids[0]],y[ids[1]],'ro')
    


class PotentialSolver(Plot):
    '''
    class to define the plate and its potential function 
    Inherits from Plot class which has all plotting functions
    '''
    def __init__(self,Nx, Ny,R ):
        '''
        Initialises the variables that is 
        R : radius 
        Nx, Ny : plate dimensions
        '''
        super(PotentialSolver, self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.radius = R
        self.X, self.Y, self.x1,self.y1 = None, None, None, None
        self.phi, self.ids = self.grid() # setting up the potential array
        self.Jx , self.Jy = None, None


    def __str__(self):
        return f'The potential array is {self.phi}'

    def grid(self):
        '''
        Function to setup the grid
        '''
        phi = np.zeros((self.Ny,self.Nx))
        self.x1 = np.linspace(-(self.Ny-1)/2,(self.Ny-1)/2,self.Ny)
        self.y1 = np.linspace(-(self.Nx-1)/2,(self.Nx-1)/2,self.Nx)
        self.Y,self.X = np.meshgrid(self.y1,self.x1)
        id = np.where((self.X**2 + self.Y**2) < self.radius*self.radius)
        phi[id] = 1.0
        return phi, id

    @staticmethod
    def step(phi, ophi):
        #phi_new = 1/4 *(left +right +top +bottom)
        phi[1:-1,1:-1] = 0.25*(ophi[1:-1,0:-2]+ophi[1:-1,2:]+ophi[0:-2,1:-1]+ophi[2:,1:-1]) 
        return phi

    @staticmethod
    def callback(phi, ids):
        '''
        function to reset the boundary conditions
        '''
        phi[1:-1,0] = phi[1:-1,1]   #left side boundary condition
        phi[1:-1,-1] = phi[1:-1,-2] #right side boundary condition
        phi[0,1:-1] = phi[1,1:-1]   #top side 
        phi[-1, 1:-1] = 0           #bottom side as its grounded
        phi[ids] = 1.0
        return phi

    def trainer(self, epochs):
        '''
        fucntion used to iterate and compute the potential
        epochs: number of times to iterate
        '''
        error = np.empty(epochs)
        for i in range(epochs):
            old_phi = self.phi.copy()
            self.phi = self.step(self.phi, old_phi)
            self.phi = self.callback(self.phi, self.ids)
            error[i] = np.max(np.abs(self.phi- old_phi)) 
        return error


    def errorFit(self,x,y):
        logy=np.log(y)
        x_arr = np.ones((len(y), 2))
        x_arr[:,0] = x
        B,logA=np.linalg.lstsq(x_arr, np.transpose(logy), rcond = None)[0]

        return np.exp(logA),B

    @staticmethod
    def max_error(A,B,N):
        return -A*(np.exp(B*(N+0.5)))/B

    @staticmethod
    def fit(x, A, B):
        return A*np.exp(B*x)
         

    def cuurents(self):
        '''
        fucntions computes the current vectors ie: Jx , Jy
        '''
        self.Jx = np.zeros((self.Ny,self.Nx))
        self.Jy = np.zeros((self.Ny,self.Nx))

        self.Jx[:,1:-1] = 0.5*(self.phi[:,0:-2]-self.phi[:,2:])
        self.Jy[1:-1,:] = 0.5*(self.phi[2:, :]-self.phi[0:-2,:])

        return self.Jx, self.Jy



if __name__ =='__main__':

    #using argparse for taking inputs as several inputs are needed and argv will get confusing
    parser = argparse.ArgumentParser()# use --help for support
    parser.add_argument('--Nx',default=25,required = True, type=int,help='Size along the x axis')
    parser.add_argument('--Ny',default=25,required = True , type=int,help='Size along the y axis')
    parser.add_argument('--radius',default=8,required = True, type=float,help='Radius of central lead')
    parser.add_argument('--Niter',default=1000,type=int,help='Number of iterations to perform', )
    args = parser.parse_args()

    # create the plate instance
    plate = PotentialSolver(args.Nx,args.Ny,args.radius)

    #Task 0
    plate.contour(plate.X, plate.Y, plate.phi, cmap=pylab.cm.get_cmap("autumn"), \
        path ='imgs/plate_plot',\
        ids = plate.ids, \
        x = plate.x1,\
        y = plate.y1)

    #compute the potential function
    loss = plate.trainer(args.Niter)
    
    #Task 1.1
    plate.semilogy(range(1000)[::50], loss[::50], \
        marker = 'ro',\
        xlabel = 'iterations',\
        ylabel = 'error(log scale)',\
        path = 'imgs/figure1',\
        label = 'plot')
    #Task 1.2
    plate.loglog(range(1000)[::50], loss[::50],\
        marker = 'ro',\
        xlabel = 'iterations(log scale)',\
        ylabel = 'error(log scale)',\
        path = 'imgs/figure2',\
        label = 'plot')

    A,B = plate.errorFit(range(args.Niter), loss)
    A2,B2 = plate.errorFit(range(args.Niter)[500:],loss[500:])

    #Task 2.1 TODO add  multiple plots cuz here u have to plot 3 in a graph
    l = range(1000)[::50]
    plate.plotMany(\
        [l]*3,\
        [loss[::50],plate.fit(l, A,B) ,plate.fit(l , A2, B2)],\
        count = 3, \
        marker = 'ro',\
        xlabel = 'iterations',\
        ylabel = 'error(log scale)',\
        path = 'imgs/figure3')

    #Task 3
    plate.plot3D(plate.X, plate.Y, plate.phi, \
        path ='imgs/figure4', 
        tile = 'The 3-D surface plot of the potential', 
        xlabel ='x',
        ylabel = 'y',
        zlabel ='phi')

    #Task 4 , contour potential
    plate.contour(plate.X, plate.Y, plate.phi, cmap=pylab.cm.get_cmap("autumn"), \
        path ='imgs/contour_plot',\
        ids = plate.ids, \
        x = plate.x1,\
        y = plate.y1 )

    #Task 5 , vetcor plot of currents 
    jx, jy = plate.cuurents()
    plate.quiver(plate.X, plate.Y, jx, jy,\
        path = 'imgs/quiver_plot',\
        x = plate.x1, \
        ids = plate.ids , \
        y = plate.y1 )













