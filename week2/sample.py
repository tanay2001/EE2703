from sys import argv
import numpy as np
import math as math
import cmath
if len(argv) != 2:                             #To ensure one input file is given
    print('\nUsage: %s <inputfile>' % argv[0]) 
    exit()
CIRCUIT = '.circuit'
END = '.end'
AC=".ac"
ac=-1  #variable to check if it is ac or dc
try:
    with open(argv[1]) as f:
        lines = f.readlines()
        start = -1; end = -2
        for line in lines:              # extracting circuit definition start and end lines
            if CIRCUIT == line[:len(CIRCUIT)]:
                start = lines.index(line)
            if END == line[:len(END)]:
                end = lines.index(line)
            if AC == line[:len(AC)]:
                ac = lines.index(line)
        
        if start >= end :# validating circuit block
            print('Invalid circuit definition')
            exit(0)
except IOError:
    print('Invalid file')
    exit()
reqd_lines = []    #required lines between start and end
freq_line=[]
[reqd_lines.append(((line.split('#')[0].split()))) for line in lines[start+1:end]]
if ac != -1:
    freq_line.append(lines[ac].split("#")[0].split())
if ac !=-1:
    freq_of_circuit = int(freq_line[0][-1])  #extracting freqency if ac circuit
class resistor(object):                       #resistor class
    def __init__(self,node1,node2,value):
        self.node1 = node1
        self.node2 = node2
        self.value = float(value)
class inductor(object):                #inductor class returns impedance as value when  value attribute is used
    if ac==-1:
        def __init__(self,node1,node2,name,value):
            self.node1 = node1
            self.node2 = node2
            self.value = 10e10
    if ac!= -1:
        def __init__(self,node1,node2,value):
            self.node1 = node1
            self.node2 = node2
            self.value = complex(0,2*math.pi*float(value)*freq_of_circuit)
class capacitor(object):             #capacitor class returns impedance as value when  value attribute is used
    if ac==-1:
        def __init__(self,node1,node2,value):
            self.node1 = node1
            self.node2 = node2
            self.value = 10e10
    if ac!= -1:
        def __init__(self,node1,node2,value):
            self.node1 = node1
            self.node2 = node2
            self.value = complex(0,(-1/(2*math.pi*float(value)*freq_of_circuit)))
class voltage(object):                #Defined voltage class
    if ac==-1:
        def __init__(self,node1,node2,name,value):
            self.node1 = node1
            self.node2 = node2
            self.value = float(value)
    if ac!= -1:
        def __init__(self,node1,node2,name,value,phase):
            self.node1 = node1
            self.node2 = node2
            self.name = name
            self.value = float(value)
            self.phase = float(phase) * (math.pi/180)
class current(object):             #Defined current class
    if ac==-1:
        def __init__(self,node1,node2,name,value):
            self.node1 = node1
            self.node2 = node2
            self.name = name
            self.value = float(value)
    if ac!= -1:
        def __init__(self,node1,node2,name,value,phase):
            self.node1 = node1
            self.node2 = node2
            self.name = name
            self.value = float(value)
            self.phase = float(phase) * (math.pi/180)
dic_of_elements={}              #Made to keep ground as first element
k=0                             #counter to count voltage sources
nodes=[]                        #list to keep track of nodes
volt_sources=[]                 #voltage sources list
for l in reqd_lines:            #for loop to dictionary of elements
    if l[0][0] == "R" :
        dic_of_elements[l[0]] = resistor(l[1],l[2],l[3])
        nodes.append(dic_of_elements[l[0]].node1)
        nodes.append(dic_of_elements[l[0]].node2)
    if l[0][0] == "L" :
        dic_of_elements[l[0]] = inductor(l[1],l[2],l[3])
        nodes.append(dic_of_elements[l[0]].node1)
        nodes.append(dic_of_elements[l[0]].node2)
        
    if l[0][0] == "C" :
        dic_of_elements[l[0]] = capacitor(l[1],l[2],l[3])
        
        nodes.append(dic_of_elements[l[0]].node1)
        nodes.append(dic_of_elements[l[0]].node2)
    if l[0][0] == "V" :
        k=k+1
        if ac==-1:
            try:
                dic_of_elements[l[0]] = voltage(l[1],l[2],l[3],l[4])
            except Exception:
                print("Expected format of dc voltage source is.......... V1 node1 node2 dc value")
        if ac!=-1:
            dic_of_elements[l[0]] = voltage(l[1],l[2],l[3],l[4],l[5])
        nodes.append(dic_of_elements[l[0]].node1)
        nodes.append(dic_of_elements[l[0]].node2)
        volt_sources.append(l[0])
    if l[0][0] == "I" :
        if ac==-1:
            try:
                dic_of_elements[l[0]] = voltage(l[1],l[2],l[3],l[4])
            except Exception:
                print("Expected format of dc current source is.......... I1 node1 node2 dc value")
        if ac!=-1:
            dic_of_elements[l[0]] = voltage(l[1],l[2],l[3],l[4],l[5])
        nodes.append(dic_of_elements[l[0]].node1)
        nodes.append(dic_of_elements[l[0]].node2)
g=dic_of_elements
nodes_list = list(set(nodes))
volt_list = list(set(volt_sources))
nodes_list.remove('GND')               #Made to keep ground as first element
nodes_final_list=['GND']+nodes_list 
k = len(volt_list)
n=len(nodes_final_list)
M = np.zeros((n+k,n+k))
y = np.zeros((n+k,1))
a={}                          # Dictionary with keys as nodes and values as elements attached to node
for i in nodes_final_list:
    a[i]=[]
    for l in reqd_lines:
        if g[l[0]].node1 ==i or g[l[0]].node2 ==i :
            a[i].append(l[0])
b={}                           #Dictionary with key as node names and values aas number given to them...number can be used as line and column in the M Matrix
d=0
for key in nodes_final_list:
    b[key]=d
    d=d+1
e={}                          #Dictionary with key as voltage sources and a number given to it!!....(n+number) can used as line number in M row
d=1
for key in volt_list:        # for loop for  node equations
    e[key]=d
    d=d+1   
for key in a:
    if b[key] == 0:
        M[b[key]][0]=1
    else:
        for i in a[key]:
            if i[0] == "R":
                if b[g[i].node1] == b[key]:
                    M[b[key]][b[g[i].node1]] += (1/g[i].value)
                    M[b[key]][b[g[i].node2]] -= (1/g[i].value)
                if b[g[i].node2] == b[key]:
                    M[b[key]][b[g[i].node2]] += (1/g[i].value)
                    M[b[key]][b[g[i].node1]] -= (1/g[i].value)
            if i[0] == "V":
                
                temp =e[i]
                if b[g[i].node1] == b[key]:
                    M[b[key]][temp+n-1] += (-1)
                if b[g[i].node2] == b[key]:
                    M[b[key]][temp+n-1] += (1)
            if i[0] =="I":
                if ac==-1:
                    if b[g[i].node1] == b[key]:
                        y[b[key]] += -g[i].value
                    if b[g[i].node2] == b[key]:
                        y[b[key]] += +g[i].value
                if ac !=-1:
                    if b[g[i].node1] == b[key]:
                        y[b[key]] += (-g[i].value)*(1/(2*math.sqrt(2)))*(cmath.rect(1,g[i].phase))
                    if b[g[i].node2] == b[key]:
                        y[b[key]] += (+g[i].value)*(1/(2*math.sqrt(2)))*(cmath.rect(1,g[i].phase))
            if i[0] =="L":
                if ac==-1:
                    print("INductor not allowed give in dc case")
                    quit()
                if ac!=-1:
                    if b[g[i].node1] == b[key]:
                        M[b[key]][b[g[i].node1]] += (1/g[i].value)
                        M[b[key]][b[g[i].node2]] -= (1/g[i].value)
                    if b[g[i].node2] == b[key]:
                        M[b[key]][b[g[i].node2]] += (1/g[i].value)
                        M[b[key]][b[g[i].node1]] -= (1/g[i].value)
            if i[0] =="C":
                if ac==-1:
                    print("Capacitor not allowed give in dc case")
                    quit()
                if ac!=-1:
                    if b[g[i].node1] == b[key]:
                        M[b[key]][b[g[i].node1]] += (1/g[i].value)
                        M[b[key]][b[g[i].node2]] -= (1/g[i].value)
                    if b[g[i].node2] == b[key]:
                        M[b[key]][b[g[i].node2]] += (1/g[i].value)
                        M[b[key]][b[g[i].node1]] -= (1/g[i].value)
                    
for c in volt_list:            #for loop for k auxilarry equations of voltage sources
    temp=e[c]
    if ac==-1:
        M[n+temp-1][b[g[c].node1]] = -1
        M[n+temp-1][b[g[c].node2]] =  1
        y[n+temp-1]=g[c].value
    if ac!=-1:
        M[n+temp-1][b[g[c].node1]] = -1
        M[n+temp-1][b[g[c].node2]] =  1
        y[n+temp-1]=(g[c].value)*(1/(2*math.sqrt(2)))*(cmath.rect(1,g[c].phase))
print(M)
x=np.linalg.solve(M, y)
i=0

print("                 RMS Values")
for k in b:
    print("VOltage at " +k + "   "+str((x[i])))
    i = i+1
for v in volt_list:
    print("Current in " +v+"  "+str((x[i])))   