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
                plt.legend()
            function(*args, **kwargs)

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
        plt.semilogy(x,y,kwargs['marker'],ms = 4, label = kwargs['label'])

    @add_stuf
    def loglog(self,x,y, **kwargs):
        plt.loglog(x,y,kwargs['marker'],ms=4, label = kwargs['label'])

        
    @add_stuf
    def plot3D(self,x,y,z, **kwargs):
        fig = plt.figure() 
        ax=p3.Axes3D(fig) 
        ax.plot_surface(y, x, z.T, rstride=1, cstride=1, cmap=plt.cm.jet)

    @add_stuf
    def plotMany(self, x, y, **kwargs):
        pass

    @add_stuf
    def contour(self, X, Y,phi, **kwargs):
        ids = kwargs.get('ids')
        x = kwargs.get('x')
        y = kwargs.get('y')
        contour(X,Y,phi)
        plot(x[ids[0]],y[ids[1]],'ro')
        #plot(ids[1]-(kwargs['Nx']-1)/2,ids[0]-(kwargs['Ny']-1)/2,kwargs['marker'])

    @add_stuf
    def quiver(self, x, y,jx, jy, **kwargs):
        inds = kwargs.get('ids')
        Nx = kwargs.get('Nx')
        Ny = kwargs.get('Ny')
        phi = kwargs.get('phi')
        fig,ax = plt.subplots()
        fig = ax.quiver(y,x,jx[::-1, :],jy[::-1, :])
        contour(x,y,phi)
        pylab.plot(inds[1]-(Nx-1)/2,inds[0]-(Ny-1)/2,'ro')
    


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
        phi = np.zeros((self.Nx,self.Ny))
        self.x1 = np.linspace(-(self.Nx-1)/2,(self.Nx-1)/2,self.Nx)
        self.y1 = np.linspace(-(self.Ny-1)/2,(self.Ny-1)/2,self.Ny)
        self.X,self.Y = np.meshgrid(self.x1,self.y1)
        id = np.where((self.X**2 + self.Y**2) <= self.radius*self.radius)
        phi[id] = 1.0
        return phi, id

    @staticmethod
    def step(phi):
        #phi_new = 1/4 *(left +right +top +bottom)
        phi[1:-1,1:-1] = 0.25*(phi[1:-1,0:-2]+phi[1:-1,2:]+phi[0:-2,1:-1]+phi[2:,1:-1]) 
        return phi

    @staticmethod
    def callback(phi, ids):
        '''
        function to reset the boundary conditions
        '''
        phi[1:-1,0] = phi[1:-1,1]   #left side boundary condition
        phi[1:-1,-1] = phi[1:-1,-2] #right side boundary condition
        #phi[0,1:-1] = phi[1,1:-1]   #top side 
        #phi[-1, 1:-1] = 0           #bottom side as its grounded
        phi[-1, 1:-1] = phi[-2, 1:-1]
        phi[0, 1:-1] =0
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
            self.phi = self.step(self.phi)
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
         

    def cuurents(self):
        '''
        fucntions computes the current vectors ie: Jx , Jy
        '''
        self.Jx = np.zeros((self.Nx,self.Ny))
        self.Jy = np.zeros((self.Nx,self.Ny))

        self.Jy[:,1:-1] = 0.5*(self.phi[:,0:-2]-self.phi[:,2:])
        self.Jx[1:-1,:] = 0.5*(self.phi[2:, :]-self.phi[0:-2,:])

        return self.Jx, self.Jy




if __name__ =='__main__':

    #using argparse for taking inputs as several inputs are needed adn argv will get confusing
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
        y = plate.y1
        )

    #compute the potential function
    loss = plate.trainer(args.Niter)

    A,B = plate.errorFit(range(args.Niter), loss)
    A2,B2 = plate.errorFit(range(args.Niter)[500:],loss[500:])

    #Task 1.1
    plate.semilogy(range(1000), loss, \
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

    #Task 2.1 TODO add  multiple plots cuz here u have to plot 3 in a graph
    plate.semilogy(range(1000)[::50],plate.max_error(A,B,np.arange(0,1000,50)),\
        marker = 'ro',\
        xlabel = 'iterations',\
        ylabel = 'error(log scale)',\
        path = 'imgs/figure3', \
        label = 'plot')

    #Task 3
    #plate.plot3D(plate.X, plate.Y, plate.phi , path ='imgs/figure4', tile ='The 3-D surface plot of the potential')

    #Task 4 , contour potential
    plate.contour(plate.X, plate.Y, plate.phi, cmap=pylab.cm.get_cmap("autumn"), \
        path ='imgs/contour_plot',\
        ids = plate.ids, \
        x = plate.x1,\
        y = plate.y1
        )

    #Task 5 , vetcor plot of currents 
    jx, jy = plate.cuurents()
    plate.quiver(plate.x1, plate.y1, jx, jy,\
        path = 'imgs/quiver_plot',\
        phi = plate.phi, \
        ids = plate.ids , \
        Nx = plate.Nx,\
        Ny = plate.Ny)













