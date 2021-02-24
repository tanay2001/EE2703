"""

EE2703 assignment 2
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: Feb 19, 2021

"""

import numpy as np
import sys
import math
import re

class Impedance():
    ##class for setting up R,L,C instances
    def __init__(self,line,ac_flag, freq=0):
        self.name, self.node1, self.node2 = line[:3]
        w = 2*np.pi*freq
        if self.name[0]=='R':
            self.value = float(line[3])
        elif self.name[0] =='C':
            C = float(line[3])
            if ac_flag:
                self.value = complex(0,-1/(w*C))
            else:
                self.value = np.inf
        elif self.name[0] =='L':
            L = float(line[3])
            if ac_flag:
                self.value = complex(0,(w*L))
            else:
                self.value = 1e-10
        else:
            print("Please follow convention in naming elemenst in circuit")
            exit()
    def __getitem__(self,key):
        pass
    
class Indsource():
    #class for setting up source instances
    def __init__(self,line, ac_flag):
            self.name, self.node1, self.node2= line[:3]
            if ac_flag:
                mag = float(line[4])/2 ## cuz Vp-p is given
                self.phase = float(line[5])* (np.pi/180)
                real = mag*np.cos(self.phase)
                img = mag*np.sin(self.phase)
                self.value = complex(real, img)
            else:
                self.value = float(line[4])
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

    return idMap

def source2id(ls):
    '''
    Takes in a list and defines the mapping of index to voltages

    @args: ls : list of all names of sources(unique)
    return a dict -->
    key: voltage names
    value: index
    '''
    #retain only Voltage sources
    vMap ={}
    c=1
    for i in ls:
        vMap[i] = c
        c+=1

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
            if n2 == node:
                circuit[node]['from'].append(l[0])
            if n1 == node:
                circuit[node]['to'].append(l[0])

    return circuit

def parseElements2obj(file, flag,f):
    '''
    converts the elements to objects
    @args : file netlist file in list format (type list[list])
    @args : flag ac flag True is ac sourcesa are used
    @args : f , freq of operation
    returns : dict, key : elements, value: class instance
    '''
    element2obj ={}
    for row in file:
        l = row.split()
        #l =[element , n1,n2, value]
        if l[0][0] in ['L','R','C']:
            t = Impedance(l,flag,f)

        elif l[0][0] in ['V','I']:
            t = Indsource(l, flag)

        element2obj[l[0]] = t

    return element2obj

def solver(node2id, source2id, element2obj,circuit, ac):
    ''''
    fills up the M , b  matrices in Mx = b
    @args node2id: type(dict) node name-->id
    @args source2id : type(dict) name--> id
    @args element2obj : type(dict) element name-->class instance
    @args circuit : type(dict) node --> elements connected to nodes
    @args ac: flag for ac sources
    '''
    n = len(node2id.keys())
    k = len(source2id.keys())
    if ac:
        M = np.zeros((n+k, n+k), dtype=complex)
        y = np.zeros((n+k,1), dtype= complex)
    else:
        M = np.zeros((n+k, n+k), dtype= np.float32)
        y = np.zeros((n+k,1), dtype = np.float32)
        
    ##fill M and y matrices and return x
    ## Mx = y
    for node in circuit:
        if node =='GND':
            #GND is set to 1 in matrix to avoid linear dependent vectors
            M[node2id[node],0] =1
        else:
            for i in circuit[node]['from']:
                #filling all from node values
                if i[0] in ['R','L','C']:
                    M[node2id[node],node2id[element2obj[i].node2]] += 1.0/element2obj[i].value
                    M[node2id[node],node2id[element2obj[i].node1]] -= 1.0/element2obj[i].value
                if ~ac:
                    if i[0] =='V':
                        M[node2id[node],source2id[i] + n -1] += 1
                    elif i[0] =='I':
                        y[node2id[node]] += (1)*element2obj[i].value
                else:
                    if i[0] =='V':
                        M[node2id[node],source2id[i] + n -1] += 1
                    elif i[0] =='I':
                        y[node2id[node]] += (1)*element2obj[i].value


            for i in circuit[node]['to']:
                #filling all to nodes values
                if i[0] in ['R','L','C']:
                    
                    M[node2id[node],node2id[element2obj[i].node1]] += 1.0/element2obj[i].value
                    M[node2id[node],node2id[element2obj[i].node2]] -= 1.0/element2obj[i].value
                if ac:
                    if i[0]=='V':
                        M[node2id[node],source2id[i] + n -1] += -1
                    elif i[0] =='I':
                        y[node2id[node]] += (-1)*element2obj[i].value
                else:
                    if i[0]=='V':
                        M[node2id[node],source2id[i] + n -1] += -1
                    elif i[0] =='I':
                        y[node2id[node]] += (-1)*element2obj[i].value


    #populate source stuff
    for source in source2id:
        M[n-1+ source2id[source], node2id[element2obj[source].node1]] = -1
        M[n-1+ source2id[source], node2id[element2obj[source].node2]] = 1
        if ~ac:
            y[n -1 + source2id[source]] = element2obj[source].value
        else:
            y[n -1 + source2id[source]] = element2obj[source].value

    return M,y

def comment_remover(lines):
    cir =[]
    ##TODO
    ### change CS case to dc/ac case print out saying currently cs course isn't supported
    for l in lines:
        l = re.sub('\n', '', l)
        ## remove comments from circuit
        try:
            i = l.index('#')
        except:
            i = len(l)
        if l[:i]!='':
            cir.append(l[:i])
    return cir

def cleaner(lines, flag, start, end, ac):
    '''
    cleans up the raw netlist file ie: removes comments
    and finds inconsistencies in the circuit definition

    @args circuit type(list[list])
    returns circuit 
    '''
    ##remove comments
    cir = comment_remover(lines)
    ##now check if V,I is represented properly
    for row in cir:
        tokens = row.split()
        if tokens[0][0] in ['R','L','C']:
            try:
                assert len(tokens) == 4, "Invalid format for impedances, follow Name n1 n2 value"
            except AssertionError as msg:
                print(msg)
                exit()
        elif tokens[0][0] in ['V','I']:
            try:
                if flag:
                    assert len(tokens) ==6 and flag, 'Inavlid format for sources follow Name n1 n2 ac value phase or Make sure .ac is present '
                else:
                    assert len(tokens) ==5, 'Inavlid format for sources follow Name n1 n2 dc value or make sure ".ac" is present if using ac source'
            except AssertionError as msg:
                print(msg)
                exit()
        elif tokens[0] in [start, end, ac]:
            continue

        elif tokens[0] in ['E','G','H','F']:
            print("sorry , controlled sources isn't supported as of such")
            exit()
        else:
            print("invalid element please use convention")
            exit()


    ##now show CSS isn't supported currently
    return cir
            
        

if __name__ =='__main__':
    try:
        assert len(sys.argv) == 2,'Please use this format : python %s <inputfile>' % sys.argv[0]
        START = '.circuit'
        END = '.end'
        AC = '.ac'
        ac_map ={}
        ac_flag = False
        try:
            with open(sys.argv[1]) as f:
                lines = f.readlines()
                c = 0
                contains = []
                freq_stuff =[]

                ##test structure of file before going to ahead
                dummy = comment_remover(lines)
                assert START in ' '.join(dummy), 'File missing .circuit'
                assert END in ' '.join(dummy), 'File missing .end'

                for l in lines:
                    tokens = l.split()
                    if START == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#'):
                        c = 1
                        continue
                    if c:
                        if END == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#'):
                            c=0
                            continue
                        assert len(l)>3,'The lines arent a valid circuit element'
                        contains.append(l)
                    
                    if AC==tokens[0]:
                        assert len(tokens)==3,'Please folow format .ac V... freq'
                        el = tokens[1]
                        freq = int(tokens[2])
                        ac_map[el] = freq
                        ac_flag =True
            f.close()
            ### cleaned up netlist file 
            circuit = cleaner(contains, ac_flag, START, END, AC)
            ### create set of nodes and set of sources
            n_list =[]; s_list =[]
            for l in circuit:
                tokens = l.split()
                element, n1,n2 = tokens[:3]
                if element[0] in ['R','L','C']:
                    n_list.append(n1)
                    n_list.append(n2)
                elif element[0] in ['V']:
                    s_list.append(element)

            assert 'GND' in n_list, 'Missing GND node please add/replace one'
            ### creating lists of unique nodes and sources
            node_set = np.unique(n_list)
            source_set = np.unique(s_list)
            ###
        
            ### creating the maps 
            n2id = node2id(node_set)
            s2id = source2id(source_set)


            nodeMap = getCircuit(circuit, node_set,)
            tokenObj = parseElements2obj(circuit, ac_flag, freq)

            ## setting up M, y and solivng the eq'n
            M , y = solver(n2id, s2id, tokenObj,nodeMap, ac_flag)
            #print(M)
            try:
                x=np.linalg.solve(M, y)
            except:
                print("matrix can't be solved")
                
            i =0
            for k in n2id:
                print("Voltage at {} is {} + j{}".format(k,x[i].real[0].astype(np.float16),\
                    x[i].imag[0].astype(np.float16)))
                i +=1
            for v in s2id:
                print("Current through {} is {} + j{}".format(v,x[i].real[0].astype(np.float16),\
                    x[i].imag[0].astype(np.float16)))
                i+=1

        except FileNotFoundError :
            print("File Not Found!!")

    except AssertionError as msg:
        print(msg)
        exit()
    
    #call app fucntions and solve circuit


        





    


































