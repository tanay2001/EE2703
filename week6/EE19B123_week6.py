'''
EE2703 assignment 6
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: April 1, 2021

'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import argparse
import mpl_toolkits.mplot3d.axes3d as p3
import os
import random
os.chdir('/home/tanay/Documents/sem4/EE2703/week6')#TODO remove this befor submitting

#########################
#creating an imgs directory to store all png files
if os.path.isdir('imgs'):
    pass
else:
    os.mkdir('imgs')
#########################
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
            if kwargs.get('legend'):
                plt.legend(kwargs['label'], loc ='upper right')
            function(*args, **kwargs)
            plt.savefig(kwargs['path']+'.png',bbox_inches='tight')
            print('File saved at {}'.format(kwargs['path']))
            plt.clf()
        return fcall

class plot:
    def __init__(self):
        pass
    @save
    def plot_intensity(self, I):
        plt.hist(I,bins=np.arange(1,100),ec='black',alpha=0.5)

    @save
    def plot_electron_density(self, X):
        plt.hist(X, bins=np.arange(1,100),ec='black',alpha=0.5)

    @save
    def plot_intensity_table(self, I):
        a,bins,c=plt.hist(I,bins=np.arange(1,100),ec='black',alpha=0.5)
        xpos=0.5*(bins[0:-1]+bins[1:])
        d={'Position':xpos,'Count':a}
        p=pd.DataFrame(data=d)
        print(p)

    @save
    def plot_electron_phase_space(self, xx, u):
        plt.plot(xx,u,'x')



class simulate(plot):
    '''

    '''
    def __init__(self,dx,x,u):
        '''

        '''
        super(simulate, self).__init__()
        self.dx  = dx # displacement at time i
        self.u  = u # e speed
        self.x = x # e postion
        self.ids = np.where(self.x > 0)[0] #places where e is present
        self.I , self.V, self.X = [], [], []


    def findElectrons(self):
        ids = np.where(self.x >0)[0]
        self.V.extend(self.u[ids].tolist())
        self.X.extend(self.x[ids].tolist())


    def displace(self):
        self.dx[self.ids] = self.u[self.ids]  +0.5
        self.x[self.ids] += self.dx[self.ids]
        self.u[self.ids] += 1

    def check_validity(self,n):
        return np.where(self.x > n)

        
    def ionization(self,p, u0):
        velocity_ids = np.where(self.u >= u0)[0]
        ll =  np.where(np.random.rand(len(velocity_ids))<=p)
        collision_ids = velocity_ids[ll]
        self.u[collision_ids] =0

        self.x[collision_ids] -= self.dx[collision_ids]*np.random.rand() #TODO improve algo
        # add photon
        self.I.extend(self.x[collision_ids].tolist())

    @staticmethod
    def injection(M, sigma):
        return np.random.randn()*sigma + M

    def inject(self,M, sigma):
        num_of_e = self.injection(M, sigma)
        empty_slots = np.where(self.x==0)[0]
        #TODO fixe error, peace out
        numOfEadded = num_of_e if num_of_e < len(empty_slots) else empty_slots

        self.x[numOfEadded] = 1
        self.u[numOfEadded] =0

    def run(self,args):
        M, sigma, p ,u0 ,n = args.M , args.Msigma, args.p , args.u0, args.n
        nk = args.nk
        for _ in range(nk):
            #dispalce the electrons
            self.displace()

            #check which have hit the anode
            hit_ids = self.check_validity(n)
            self.u[hit_ids] =0
            self.x[hit_ids] =0
            self.dx[hit_ids] =0

            #ionize the electrons that have hit the anode
            self.ionization(p , u0)

            #inject more electrons
            self.inject(M, sigma)

            #identify elect properties
            self.findElectrons() #TODO optimise by ids + numOfEadded - hits(ll)

        return self.I, self.V , self.X

            


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

    chamber = simulate(dx,x,u)

    I, V, X = chamber.run(args)

    chamber.plot_electron_density(X, path = 'electron_density')
    chamber.plot_electron_phase_space(chamber.x , chamber.u, path ='phase+plot')
    chamber.plot_intensity(I, path = 'plot_intensity')
    chamber.plot_intensity_table(I, path = 'plot-table')

