'''
EE2703 assignment 5
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: March 17, 2021

'''

import numpy as np 
import matplotlib.pyplot as plt 
import argparse
import mpl_toolkits.mplot3d.axes3d as p3




class Plot():
    def __init__(self):
        '''
        '''
        pass

    @staticmethod
    def add_stuff(func):
        def stuff(*args, **kwargs):
            plt.xlabel(kwargs.xlabel)
            plt.ylabel(kwargs.ylabel)
            plt.title(kwargs.title)
            if kwargs.legend:
                plt.legend()
            
        return stuff

    @classmethod
    @add_stuff
    def semilogy(cls,x,y, **kwargs):
        plt.semilogy(x,y, marker = kwargs.marker, label = kwargs.label)
        plt.savefig(kwargs.path+'.png',bbox_inches='tight')

    @classmethod
    @add_stuff
    def loglog(cls,x,y, **kwargs):
        plt.semilogy(x,y, marker = kwargs.marker, label = kwargs.label)
        plt.savefig(kwargs.path+'.png',bbox_inches='tight')

    @classmethod
    @add_stuff
    def plot3D(cls,x,y, **kwargs):
        plt.semilogy(x,y, marker = kwargs.marker, label = kwargs.label)
        plt.savefig(kwargs.path+'.png',bbox_inches='tight')

    

class PotentialSolver(Plot):
    def __init__(self,Nx, Ny,R ):
        '''
        '''
        super(PotentialSolver, self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.radius = R
        self.phi, self.ids = self.grid()


        # intilise the array use a class attribute
        # 
    def __str__(self):
        return f'The potential array is {self.phi}'
         

    def grid(self):
        phi = np.zeros((self.Ny,self.Nx))
        x1 = np.linspace(-(self.Ny-1)/2,(self.Ny-1)/2,self.Ny)
        y1 = np.linspace(-(self.Nx-1)/2,(self.Nx-1)/2,self.Nx)
        Y,X = np.meshgrid(y1,x1)
        id = np.where((X**2 + Y**2) <= ((self.radius*self.Nx)/100)**2)
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


    def trainer(self, epochs):
        error = np.empty(epochs)
        for i in range(epochs):
            old_phi = self.phi.copy()
            self.phi = self.step(self.phi, old_phi)
            self.phi = self.callback(self.phi, self.ids)
            error[i] = np.max(np.abs(self.phi- old_phi))
            
            # 

if __name__ =='__main__':
    parser = argparse.ArgumentParser()# use --help for support
    parser.add_argument('--Nx',default=25,required = True, type=int,help='Size along the x axis')
    parser.add_argument('--Ny',default=25,required = True , type=int,help='Size along the y axis')
    parser.add_argument('--radius',default=8,required = True, type=float,help='Radius of central lead')
    parser.add_argument('--Niter',default=1000,type=int,help='Number of iterations to perform', )
    args = parser.parse_args()

    plate = PotentialSolver(25,25,8)

    plate.trainer(1000)

    plate.semilogy()