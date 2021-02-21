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
            self.name, self.node1, self.node2, self.element = line[:4]
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

def getTokens(line, nodes_list):
    '''
    Finds all the elements attached to respective nodes

    @args line: line of the netlist file (type list)
    @args nodes_list: set of unique nodes in the circuit (type set)
    returns a dict
    keys: nodes
    values : list of elements either having the node as "from" or "to" 
    '''
    tokens = line.split()
    l = len(tokens)
    


































