"""
EE2703 assignment 2
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: Feb 19, 2021
"""
import numpy as np
import sys

class Impedance():
    def __init__(self,line, w=0):
        self.name, self.node1, self.node2 = line[:3]
        if self.name[0]=='R':
            self.value = line[3]
        elif self.name[0] =='C':
            self.value = -j/(w*line[3])
        elif self.name[0] =='L':
            self.value = (w*line[3])*j
        else:
            print("Please follow convention")
            exit()
    def __getitem__(self,key):
        pass
    
class Indsource():
    def __init__(self,line):
            self.name, self.node1, self.node2, self.value = line[:4]
    def __getitem__(self,key):
        pass

def node2id(ls):
    '''
    Takes in a list and defines the mapping of index to names
    @args: ls : list of all names of nodes(unique)
    return a dict -->
    key: name
    value: index
    '''
    idMap ={'GND':0}
    c=1
    ls.remove('GND')
    for i in ls:
        idMap[i] = c
        c+=1
    #unit test
    assert idMap.keys() != None

    return idMap

def source2id(ls):
    '''
    Takes in a list and defines the mapping of index to voltages

    @args: ls : list of all names of voltages(unique)
    return a dict -->
    key: voltage names
    value: index
    '''
    vMap ={}
    c=0
    for i in ls:
        vMap[i] = c
        c+=1
    #unit test
    assert vMap.keys() != None

    return vMap

def getCircuit(file, nodes_list):
    '''
    Finds all the elements attached to respective nodes

    @args file: netlist file in list format (type list[list])
    @args nodes_list: set of unique nodes in the circuit (type set)
    returns a dict
    keys: nodes
    values : list of elements either having the node as "from" or "to" 
    '''
    circuit ={}
    for node in nodes_list:
        circuit[node] =[]
        for l in file:
            #l : [name, n1,n2,value]
            n1 = l[1]
            n2 = l[2]
            if n1 == node or n2 == node:
                circuit[node].append(l[0])

    return circuit

def parseElements2obj(file):
    '''
    converts the elements to objects
    @args : file netlist file in list format (type list[list])
    returns : dict, key : elements, value: class instance
    '''
    element2obj ={}
    for row in file:
        #row =[element , n1,n2, value]
        if row[0][0] in ['L','R','C']:
            t = Impedance(row)

        elif row[0][0] in ['V','I']:
            t = Indsource(row)

        element2obj[row[0]] =t
    
    return element2obj

def solver(node2id, source2id, element2obj,circuit):
    ''''
    fills up the M  matrix in Mx = b
    @args node2id: type(dict) node name-->id
    @args source2id : type(dict) name--> id
    @args element2obj : type(dict) element name-->class instance
    @args circuit : type(dict) node --> elements connected to nodes
    '''
    M = np.empty((n+k, n+k))
    y = np.empty((n+k,1))
    ##fill M and y matrices and return x
    ## Mx = y





    return M

if __name__ =='__main__':
    #remove comments from netlist file
    # create a cleam list[list] file 
    #call app fucntions and solve circuit


        





    


































