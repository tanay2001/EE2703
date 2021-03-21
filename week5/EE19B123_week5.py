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
os.chdir('/home/tanay/Documents/sem4/EE2703/week5')

##TODO make imgs directory



#python decorator to add plot label/title/legend/save file
def add_stuf(function):
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
        return fcall


class Plot:
    '''
    TODO add description
    '''

    def __init__(self):
        super(Plot, self).__init__()


    @add_stuf
    def semilogy(self,x,y, **kwargs):
        plt.semilogy(x,y,kwargs['marker'],ms = 4, label = kwargs['label'])
        print('File saved at {}'.format(kwargs['path']))
        plt.clf()

    @add_stuf
    def loglog(self,x,y, **kwargs):
        plt.loglog(x,y,kwargs['marker'],ms=4, label = kwargs['label'])
        print('File saved at {}'.format(kwargs['path']))
        plt.clf()

    @add_stuf
    def plot(self,x,y, **kwargs):
        plt.plot(x,y,kwargs['marker'],ms=4, label = kwargs['label'])
        print('File saved at {}'.format(kwargs['path']))
        plt.clf()

    @add_stuf
    def plot3D(self,x,y,z, **kwargs):
        fig = plt.figure() 
        ax=p3.Axes3D(fig) 
        ax.plot_surface(y, x, z.T, rstride=1, cstride=1, cmap=plt.cm.jet)

    @add_stuf
    def plotMany(self, x, y, **kwargs):
        pass

    

class PotentialSolver(Plot):
    '''
    TODO add description
    '''
    def __init__(self,Nx, Ny,R ):
        '''

        '''
        super(PotentialSolver, self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.radius = R
        self.phi, self.ids = self.grid()
        self.X, self.Y = None, None


    def __str__(self):
        return f'The potential array is {self.phi}'


    def errorFit(self,x,y):
        logy=np.log(y)
        x_arr = np.ones((len(y), 2))
        x_arr[:,0] = x
        B,logA=np.linalg.lstsq(x_arr, np.transpose(logy))[0]

        return np.exp(logA),B

    @staticmethod
    def max_error(A,B,N):
        return -A*(np.exp(B*(N+0.5)))/B
         

    def grid(self):
        phi = np.zeros((self.Ny,self.Nx))
        x1 = np.linspace(-(self.Ny-1)/2,(self.Ny-1)/2,self.Ny)
        y1 = np.linspace(-(self.Nx-1)/2,(self.Nx-1)/2,self.Nx)
        self.Y,self.X = np.meshgrid(y1,x1)
        id = np.where((self.X**2 + self.Y**2) <= ((self.radius*self.Nx)/100)**2)
        phi[id] = 1.0
        return phi, id

    @staticmethod
    def step(phi_new, phi_old):
        #phi_new = 1/4 *  (               left +              right +             top +           bottom)
        phi_new[1:-1,1:-1] = 0.25*(phi_old[1:-1,0:-2]+phi_old[1:-1,2:]+phi_old[0:-2,1:-1]+phi_old[2:,1:-1]) 
        return phi_new

    @staticmethod
    def callback(phi, ids):
        phi[1:-1,0] = phi[1:-1,1] #left side boundary condition
        phi[1:-1,-2] = phi[1:-1,-1] #right side boundary condition
        phi[0,1:-1] = phi[1,1:-1]    #top side 
        phi[-1, 1:-1] = 0 # bottom side as its grounded
        phi[ids] = 1.0

        return phi


    def trainer(self, epochs):
        error = np.empty(epochs)
        for i in range(epochs):
            old_phi = self.phi.copy()
            self.phi = self.step(self.phi, old_phi)
            self.phi = self.callback(self.phi, self.ids)
            error[i] = np.max(np.abs(self.phi- old_phi))
            
        return error



if __name__ =='__main__':
    parser = argparse.ArgumentParser()# use --help for support
    parser.add_argument('--Nx',default=25,required = True, type=int,help='Size along the x axis')
    parser.add_argument('--Ny',default=25,required = True , type=int,help='Size along the y axis')
    parser.add_argument('--radius',default=8,required = True, type=float,help='Radius of central lead')
    parser.add_argument('--Niter',default=1000,type=int,help='Number of iterations to perform', )
    args = parser.parse_args()

    plate = PotentialSolver(25,25,8)

    loss = plate.trainer(1000)

    A,B = plate.errorFit(range(1000), loss)
    A2,B2 = plate.errorFit(range(1000)[500:],loss[500:])

    #Task 1.1
    plate.semilogy(range(1000), loss, marker = 'ro', xlabel = 'iterations', ylabel = 'error)log scale)',path = 'imgs/figure1', label = 'plot')
    #Task 1.2
    plate.loglog(range(1000)[::50], loss[::50], marker = 'ro', xlabel = 'iterations(log scale)', ylabel = 'error(log scale)',path = 'imgs/figure2', label = 'plot')

    #Task 2.1 TODO add  multiple plots cuz here u have to plot 3 in 2 graph
    plate.semilogy(range(1000)[::50],plate.max_error(A,B,np.arange(0,1000,50)), marker = 'ro', xlabel = 'iterations', ylabel = 'error)log scale)',path = 'imgs/figure3', label = 'plot')

    #Task 3
    plate.plot3D(plate.X, plate.Y, plate.phi , path ='imgs/figure4', tile ='The 3-D surface plot of the potential')

    #Task 4 , contour potential

    #Task 5 , vetcor plot of currents 













