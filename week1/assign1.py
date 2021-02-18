
"""
EE2703 assignment 1
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: Feb 13, 2021
"""

import sys
import string

def getTokens(line):
    '''
    Takes in each line of input and prints tokens
    @args : line of the input file
    returns None
    '''
    tokens = line.split()
    l = len(tokens)
    
    if l==4 or (l>4 and tokens[4][0] =='#'):
        ### l =4 implies impedance element
        ### but incase a comment is present it may misleed the element type
        ### hence total tokens apart from comments should be 4
        element,n1,n2,value = tokens[:4]

        ### confirming if all values are alphanumeric
        try:
            assert  n1.isalnum() and n2.isalnum(), "Node names need to be alphanumeric, please check Input file row"

            if element[0]=='R':
                t ='Resistor'
            elif element[0]=='C':
                t = 'Capacitor'
            elif element[0]=='L':
                t = 'Inductor'
            elif element[0]=='V':
                t = 'Ind. voltage source'
            else:
                t = 'Ind. current source'
            
            print(value,n2,n1,element)
            #print('{}({}) of value {} connected from {} to {}'.format(t,element,value,n2,n1) )
        except AssertionError as msg:  
            print(msg)

    elif l == 6 or (l>6 and tokens[6][0] =='#'):
        ### l =6 implies VCVS/VCCS source
        ### but incase a comment is present it may misleed the element type
        ### hence total tokens apart from comments should be 6
        element,n1,n2,n3,n4,value = tokens[:6]

        ### confirming if all values are alphanumeric
        try:
            assert n1.isalnum() and n2.isalnum() and n3.isalnum() and n4.isalnum(), "Node names need to be alphanumeric, please check Input file row"
            if element[0] =='E':
                t='VCVS'
            elif element[0]=='G':
                t = 'VCCS'
            else:
                t = element
            print(value,n4,n3,n2,n1,element)
            #print('{}({}) of value {} , connected from {} to {} and is  dependent on controlled source across {} and {}'.format(t,element, value,n2,n1,n4,n3))

        except AssertionError as msg:  
            print(msg)

    elif l==5 or (l>5 and tokens[5][0] =='#'):
        ### l =5 implies CCVS/CCCS source
        ### but incase a comment is present it may mislead the element type
        ### hence total tokens apart from comments should be 5
        element,n1,n2,V,value = tokens[:5]

        ### confirming if all values are alphanumeric
        try:
            assert n1.isalnum() and n2.isalnum(),"Node names are alphanumeric, please check Input file row"
            if element[0]=='H':
                t ='CCVS'
            elif element[0]=='F':
                t ='CCCS'
            else:
                t = element
            print(value,V,n2,n1,element)
            #print('{}({}) of value {} , connected from {} to {} and is  dependent on current through voltgae source {}'.format(t,element, value,n2,n1,V))
        except AssertionError as msg:  
            print(msg)

if __name__ =='__main__':
    try:
        assert len(sys.argv) == 2,'Please use this format : python %s <inputfile>' % sys.argv[0]
        START = '.circuit'
        END = '.end'
        try:
            with open(sys.argv[1]) as f:
                lines = f.readlines()
                c = 0
                contains = []
                for l in reversed(lines):
                    tokens = l.split()
                    if END == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#'):
                        c = 1
                        continue
                    if c:
                        if START == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#'):
                            break
                        assert len(l)>3,'The lines arent a valid circuit element'
                        contains.append(getTokens(l))
            if contains ==[]:
                print('No circuit block found check input file for .end and .circuit lines')
            f.close()
    
        except FileNotFoundError :
            print("File Not Found!!")

    except AssertionError as msg:
        print(msg)