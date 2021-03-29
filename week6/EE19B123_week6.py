'''
EE2703 assignment 6
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: April 1, 2021

'''
import numpy as np 
import matplotlib.pyplot as plt 
import argparse
import mpl_toolkits.mplot3d.axes3d as p3
import os
from pylab import contour, plot, contourf, cm
import random
os.chdir('/home/tanay/Documents/sem4/EE2703/week6')#TODO remove this befor submitting

#########################
#creating an imgs directory to store all png files
if os.path.isdir('imgs'):
    pass
else:
    os.mkdir('imgs')
#########################

class simulate:
    '''

    '''
    def __init__(self,dx, ids, x,u, args):
        '''

        '''
        self.dx  = dx # displacement at time i
        self.ids = ids #places where e is present
        self.u  = u # e speed
        self.x = x # e postion
        self.I , self.V, self.X = [], [], []

    def displace(self):
        self.dx[self.ids] = self.u[self.ids]  +0.5
        self.x[self.ids] += self.dx[self.ids]
        self.u[self.ids] += 1

    def check_validity(self):
        hit_ids = np.where(self.x > n)
        self.u[hit_ids] =0
        self.x[hit_ids] =0
        self.dx[hit_ids] =0

    def threshold(self,p, u0):
        velocity_ids = np.where(self.u >= u0)
        ll =  np.where(np.random.rand(len(velocity_ids[0]))<=p)
        collision_ids = velocity_ids[ll]
        self.u[collision_ids] =0

        self.x[collision_ids] -= self.dx[collision_ids]*np.random.rand() #TODO improve algo
        # add photon
        self.I.extend(self.x[collision_ids].tolist())

    @staticmethod
    def injection(M, sigma , mean):
        return np.random.randn()*sigma + mean

    







if __name__ =='__main__':
    parser = argparse.ArgumentParser()# use --help for support
    parser.add_argument('--n',default=100,required = True, type=int,help='spatial grid size')
    parser.add_argument('--M',default=5,required = True , type=int,help='number of electrons injected per turn')
    parser.add_argument('--nk',default=500,required = True, type=int,help='number of turns to simulate')
    parser.add_argument('--u0',default=5,required = True ,type=float,help='threshold velocity')
    parser.add_argument('--p',default=0.25,required = True ,type=float,help='probability that ionization will occur')
    parser.add_argument('--Msigma',default=0.2,required = True ,type=float,help='std in number of electrons added')
    args = parser.parse_args()

    x , u, dx = np.zeros((1,args.n*args.M)), np.zeros((1,args.n*args.M)),np.zeros((1,args.n*args.M))

