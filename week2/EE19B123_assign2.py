"""
EE2703 assignment 2
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: Feb 19, 2021
"""
import numpy as np
import sys
import math

class Impedance():
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
                self.value = 0
        else:
            print("Please follow convention in naming elemenst in circuit")
            exit()
    def __getitem__(self,key):
        pass
    
class Indsource():
    def __init__(self,line, ac_flag):
            self.name, self.node1, self.node2= line[:3]
            if ac_flag:
                self.value = float(line[4])
                self.phase = float(line[5])* (np.pi/180)
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
    assert idMap.keys() != None

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

def parseElements2obj(file, flag):
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
            t = Impedance(l,flag)

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
    '''
    n = len(node2id.keys())
    k = len(source2id.keys())
    if ac:
        M = np.zeros((n+k, n+k), dtype=complex)
        y = np.zeros((n+k,1), dtype= complex)
    else:
        M = np.zeros((n+k, n+k))
        y = np.zeros((n+k,1))
        
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
                elif i[0] =='V':
                    M[node2id[node],source2id[i] + n -1] += 1
                elif i[0] =='I':
                    y[node2id[node]] += (1)*element2obj[i].value

            for i in circuit[node]['to']:
                #filling all to nodes values
                if i[0] in ['R','L','C']:
                    M[node2id[node],node2id[element2obj[i].node1]] += 1.0/element2obj[i].value
                    M[node2id[node],node2id[element2obj[i].node2]] -= 1.0/element2obj[i].value
                elif i[0]=='V':
                    M[node2id[node],source2id[i] + n -1] += -1
                elif i[0] =='I':
                    y[node2id[node]] += (-1)*element2obj[i].value

    #populate source stuff
    for source in source2id:
        M[n-1+ source2id[source], node2id[element2obj[source].node1]] = -1
        M[n-1+ source2id[source], node2id[element2obj[source].node2]] = 1

        y[n -1 + source2id[source]] = element2obj[source].value

    return M,y

def cleaner(lines):
    '''
    cleans up the raw netlist file ie: removes comments
    and finds inconsistencies in th circuit definition

    @args circuit type(list[list])
    returns circuit 
    '''
    cir =[]
    for l in lines:
        tokens = l.split()
        length = len(tokens)
        if length==4 or (length>4 and tokens[4][0] =='#'):
            ### l =4 implies impedance element
            ### but incase a comment is present it may misleed the element type
            ### hence total tokens apart from comments should be 4
            element,n1,n2,value = tokens[:4]

            ### confirming if all values are alphanumeric
            try:
                assert  n1.isalnum() and n2.isalnum(), "Node names need to be alphanumeric, please check Input file row"

            except AssertionError as msg:  
                print(msg)

        elif length == 6 or (length>6 and tokens[6][0] =='#'):
            ### l =6 implies VCVS/VCCS source
            ### but incase a comment is present it may misleed the element type
            ### hence total tokens apart from comments should be 6
            element,n1,n2,n3,n4,value = tokens[:6]

            ### confirming if all values are alphanumeric
            try:
                assert n1.isalnum() and n2.isalnum() and n3.isalnum() and n4.isalnum(), "Node names need to be alphanumeric, please check Input file row"
            except AssertionError as msg:  
                print(msg)

        elif length==5 or (length>5 and tokens[5][0] =='#'):
            ### l =5 implies CCVS/CCCS source
            ### but incase a comment is present it may mislead the element type
            ### hence total tokens apart from comments should be 5
            element,n1,n2,V,value = tokens[:5]

            ### confirming if all values are alphanumeric
            try:
                assert n1.isalnum() and n2.isalnum(),"Node names are alphanumeric, please check Input file row"
            except AssertionError as msg:  
                print(msg)


        ## remove comments from circuit
        try:
            i = l.index('#')
        except:
            i = len(l)
        cir.append(l[:i])
    return cir
            
        

if __name__ =='__main__':
    try:
        assert len(sys.argv) == 2,'Please use this format : python %s <inputfile>' % sys.argv[0]
        START = '.circuit'
        END = '.end'
        AC = '.ac'
        ac_flag = False
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
                            c=0
                            continue
                        assert len(l)>3,'The lines arent a valid circuit element'
                        contains.append(l)
                    if AC==tokens[0] and (len(tokens)==1 or tokens[1][0] =='#'):
                        freq = int(tokens[1])
                        ac_flag =True
            f.close()
            ### cleaned up netlist file 
            circuit = cleaner(contains)
            
            ### create set of nodes and set of sources
            n_list =[]; s_list =[]
            for l in circuit:
                tokens = l.split()
                element, n1,n2,value = tokens[:4]
                if element[0] in ['R','L','C']:
                    n_list.append(n1)
                    n_list.append(n2)
                elif element[0] in ['V']:
                    s_list.append(element)

            ### creating lists of unique nodes and sources
            node_set = np.unique(n_list)
            source_set = np.unique(s_list)
            ###

            ### creating the maps 
            n2id = node2id(node_set)
            s2id = source2id(source_set)

            nodeMap = getCircuit(circuit, node_set,)
            tokenObj = parseElements2obj(circuit, ac_flag)

            ## setting up M, y and solivng the eq'n
            M , y = solver(n2id, s2id, tokenObj,nodeMap, ac_flag)
            print(M)
            try:
                x=np.linalg.solve(M, y)
            except:
                print("matrix can't be solved")
                
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


        





    


































