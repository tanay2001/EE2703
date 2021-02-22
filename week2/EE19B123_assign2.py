"""
EE2703 assignment 2
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: Feb 19, 2021
"""
import numpy as np
import sys
import math

class Impedance():
    def __init__(self,line, w=0):
        self.name, self.node1, self.node2 = line[:3]
        if self.name[0]=='R':
            self.value = float(line[3])
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
            self.name, self.node1, self.node2= line[:3]
            self.value = float(line[3])
    def __getitem__(self,key):
        pass

def node2id(l):
    '''
    Takes in a list and defines the mapping of index to names
    @args: ls : list of all names of nodes(unique)
    return a dict -->
    key: name
    value: index
    '''
    idMap ={'GND':0}
    c=1
    ls = list(l)
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

    @args:= file: netlist file in list format (type list[list])
    @args:= nodes_list: set of unique nodes in the circuit (type list)
    returns a dict
    keys: nodes
    values : list of elements either having the node as "from" or "to" 
    '''
    circuit ={}
    for node in nodes_list:
        circuit[node] ={'from':[], 'to':[]}
        for t in file:
            l = t.split()
            #l : [name, n1,n2,value]
            n1 = l[1]
            n2 = l[2]
            if n1 == node:
                circuit[node]['from'].append(l[0])
            if n2 == node:
                circuit[node]['to'].append(l[0])

    return circuit

def parseElements2obj(file):
    '''
    converts the elements to objects
    @args : file netlist file in list format (type list[list])
    returns : dict, key : elements, value: class instance
    '''
    element2obj ={}
    for row in file:
        l = row.split()
        #l =[element , n1,n2, value]
        if l[0][0] in ['L','R','C']:
            t = Impedance(l)

        elif l[0][0] in ['V','I']:
            t = Indsource(row)

        element2obj[l[0]] =t
    
    return element2obj

def solver(node2id, source2id, element2obj,circuit):
    ''''
    fills up the M , b  matrices in Mx = b
    @args node2id: type(dict) node name-->id
    @args source2id : type(dict) name--> id
    @args element2obj : type(dict) element name-->class instance
    @args circuit : type(dict) node --> elements connected to nodes
    '''
    n = len(node2id.keys())
    k = len(source2id.keys())

    M = np.zeros((n+k, n+k))
    y = np.zeros((n+k,1))
    ##fill M and y matrices and return x
    ## Mx = y
    print(element2obj)
    for node in circuit:
        if node =='GND':
            #GND is set to 1 in matrix to avoid linear dependent vectors
            M[node2id[node],0] =1
        else:
            for i in circuit[node]['from']:
                #filling all from node values
                if i[0] in ['R','L','C']:
                    M[node2id[node],node2id[element2obj[i].node1]] += 1.0/element2obj[i].value
                    M[node2id[node],node2id[element2obj[i].node2]] -= 1.0/element2obj[i].value
                elif i[0] =='V':
                    M[node2id[node],source2id[i] + n -1] += -1
                elif i[0] =='I':
                    y[node2id[node]] += (-1)*element2obj[i].value

            for i in circuit[node]['to']:
                #filling all to nodes values
                if i[0] in ['R','L','C']:
                    M[node2id[node],node2id[element2obj[i].node2]] += 1.0/element2obj[i].value
                    M[node2id[node],node2id[element2obj[i].node1]] -= 1.0/element2obj[i].value
                elif i[0]=='V':
                    M[node2id[node],source2id[i] + n -1] += 1
                elif i[0] =='I':
                    y[node2id[node]] += element2obj[i].value

    #populate source stuff
    for source in source2id:
        M[n+ source2id[source], node2id[element2obj[source].node1]] = -1
        M[n+ source2id[source], node2id[element2obj[source].node2]] = 1

        y[n -1 + source2id[source]] = element2obj[source].value

    return M,y

def cleaner(lines):
    '''
    cleans up the raw netlist file ie: removes comments and converts str to int where needed
    @args circuit type(list[list])
    returns circuit 
    '''
    cir =[]
    for l in lines:
        try:
            i = l.index('#')
        except:
            i = len(l)
        cir.append(l[:i])
    return cir
            
        

if __name__ =='__main__':
    #remove comments from netlist file
    try:
        assert len(sys.argv) == 2,'Please use this format : python %s <inputfile>' % sys.argv[0]
        START = '.circuit'
        END = '.end'
        try:
            with open(sys.argv[1]) as f:
                lines = f.readlines()
                c = 0
                contains = []
                for l in lines:
                    tokens = l.split()
                    if START == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#'):
                        c = 1
                        continue
                    if c:
                        if END == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#'):
                            break
                        assert len(l)>3,'The lines arent a valid circuit element'
                        contains.append(l)
            f.close()
            ### cleaned up netlist file 
            circuit = cleaner(contains)
            print(circuit)
            ### create set of nodes and set of voltages
            n_list =[]; s_list =[]
            for l in circuit:
                tokens = l.split()
                element, n1,n2,value = tokens[:4]
                if element[0] in ['R','L','C']:
                    n_list.append(n1)
                    n_list.append(n2)
                elif element[0] in ['V','I']:
                    s_list.append(element)

            node_set = np.unique(n_list)
            source_set = np.unique(s_list)
            n2id = node2id(node_set)
            s2id = source2id(source_set)

            nodeMap = getCircuit(circuit, node_set,)
            tokenObj = parseElements2obj(circuit)
            M , y = solver(n2id, s2id, tokenObj,nodeMap)

            x=np.linalg.solve(M, y)

            i =0
            for k in n2id:
                print("Voltage at {} is {}".format(k,x[i]))
                i = i+1
            for v in s2id:
                print("Current through {} is {}".format(k,x[i]))

            print(x)

    
        except FileNotFoundError :
            print("File Not Found!!")

    except AssertionError as msg:
        print(msg)
    
    #call app fucntions and solve circuit


        





    


































