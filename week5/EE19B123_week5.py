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
    @classmethod
    def semilogy(x,y, xlabel, ylabel, title ,marker, legend = False, grid = True ):
        pass

    @classmethod
    def loglog(x,y, xlabel, ylabel, title ,marker, legend = False, grid = True ):
        pass

class PotentialSolver(Plot):
    def __init__(self,Nx, Ny,R ):
        '''
        '''
        super(PotentialSolver, self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.radius = R
        self.phi = self.grid()

        # intilise the array use a class attribute
        # 
    def __str__(self):
        return 'The potential array is {}'.format(self.phi)
         

    def grid(self):
        phi = np.zeros((self.Ny,self.Nx))
        x1 = np.linspace(-(self.Ny-1)/2,(self.Ny-1)/2,self.Ny)
        y1 = np.linspace(-(self.Nx-1)/2,(self.Nx-1)/2,self.Nx)
        Y,X = np.meshgrid(y1,x1)
        id = np.where((X**2 + Y**2) <= ((self.radius*self.Nx)/100)**2)
        phi[id] = 1
        return phi 

    @staticmethod
    def step(phi_new, phi_old):
        phi_new[1:-1,1:-1] = 0.25*(phi_old[1:-1,0:-2]+phi_old[1:-1,2:]+phi_old[0:-2,1:-1]+phi_old[2:,1:-1]) 
        return phi_new

    @staticmethod
    def callback(phi):
        pass


    @classmethod
    def trainer(cls,phi, epochs):
        error = np.empty(epochs)
        for i in range(epochs):
            old_phi = phi.copy()
            phi = step(phi, old_phi)
            phi = callback(phi)
            error[i] = np.max(np.abs(phi- old_phi))
            
            # 



        return cls 







        

if __name__ =='__main__':
    parser = argparse.ArgumentParser()# use --help for support
    parser.add_argument('--Nx',default=25,required = True, type=int,help='Size along the x axis')
    parser.add_argument('--Ny',default=25,required = True , type=int,help='Size along the y axis')
    parser.add_argument('--radius',default=8,required = True, type=float,help='Radius of central lead')
    parser.add_argument('--Niter',default=1000,type=int,help='Number of iterations to perform', )
    args = parser.parse_args()