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
import gc
from tqdm import tqdm
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
    def plotIntensity(self, I, **kwargs):
        plt.hist(I,bins=np.arange(1,100),ec='black',alpha=0.5)

    @save
    def plotEdensity(self, X, **kwargs):
        plt.hist(X, bins=np.arange(1,100),ec='black',alpha=0.5)

    @save
    def table(self, I, **kwargs):
        a,bins,c=plt.hist(I,bins=np.arange(1,100),ec='black',alpha=0.5)
        xpos=0.5*(bins[0:-1]+bins[1:])
        d={'Position':xpos,'Count':a}
        p=pd.DataFrame(data=d)
        print(p)

    @save
    def plot_electron_phase_space(self, xx, u, **kwargs):
        plt.plot(xx,u,'x')



class simulate(plot):
    '''
    enviornment for Tubelight Simulation
    '''
    def __init__(self,dx,x,u):
        '''
        @Params:
        dx: current displacement
        x : electron position
        u : electron velocity

        '''
        super(simulate, self).__init__()
        self.dx  = dx # displacement at time i
        self.u  = u # e speed
        self.x = x # e postion
        self.I , self.V, self.X = [], [], []


    def findElectrons(self):
        ids = np.where(self.x >0)[0]
        self.V.extend(self.u[ids].tolist())
        self.X.extend(self.x[ids].tolist())


    def displace(self):
        self.ids = np.where(self.x > 0)[0]
        self.dx[self.ids] = self.u[self.ids]  +0.5
        self.x[self.ids] += self.dx[self.ids]
        self.u[self.ids] += 1

    def check_validity(self,n):
        return np.where(self.x > n)

        
    def ionization(self,p, u0):
        velocity_ids = np.where(self.u >= u0)[0]
        ll =  np.where(np.random.rand(len(velocity_ids))<=p)[0]
        collision_ids = velocity_ids[ll]
        self.u[collision_ids] =0

        self.x[collision_ids] -= self.dx[collision_ids]*np.random.rand() #TODO improve algo
        # add photon
        self.I.extend(self.x[collision_ids].tolist())
        #print('#Electrons Ionized')

    @staticmethod
    def injection(M, sigma):
        return np.random.randn()*sigma + M

    def inject(self,M, sigma):
        num_of_e = int(self.injection(M, sigma))
        empty_slots = np.where(self.x==0)[0]
        numOfEadded = random.sample(empty_slots.tolist(),num_of_e) if num_of_e <= len(empty_slots) else empty_slots

        self.x[numOfEadded] = 1
        self.u[numOfEadded] =0
        #print('##Electrons added')

    def run(self,args):
        M, sigma, p ,u0 ,n = args.M , args.Msigma, args.p , args.u0, args.n
        nk = args.nk
        for _ in tqdm(range(nk)): #range(nk)
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

        print(len(self.I), len(self.V), len(self.X))
        return self.I, self.V , self.X

            
if __name__ =='__main__':
    parser = argparse.ArgumentParser()# use --help for support
    #TODO add required
    parser.add_argument('--n',default=100, type=int,help='spatial grid size')
    parser.add_argument('--M',default=5 , type=int,help='number of electrons injected per turn')
    parser.add_argument('--nk',default=500, type=int,help='number of turns to simulate')
    parser.add_argument('--u0',default=5,type=float,help='threshold velocity')
    parser.add_argument('--p',default=0.25,type=float,help='probability that ionization will occur')
    parser.add_argument('--Msigma',default=0.2,type=float,help='std in number of electrons added')
    args = parser.parse_args()

    x , u, dx = np.zeros(args.n*args.M), np.zeros(args.n*args.M),np.zeros(args.n*args.M)

    #creating the environment
    chamber = simulate(dx,x,u)

    I, V, X = chamber.run(args)
    print('done running')
    gc.collect()

    chamber.plotEdensity(X, path = 'imgs/electron_density')
    chamber.plot_electron_phase_space(chamber.x , chamber.u, path ='imgs/phase+plot')
    chamber.plotIntensity(I, path = 'imgs/plot_intensity')
    chamber.table(I, path = 'imgs/plot-table')

