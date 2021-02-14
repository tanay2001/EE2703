# -*- coding: utf-8 -*-
"""
EE2703 assignment 1
Author: Tanay Dixit <ee19b123@smail.iitm.ac.in>
Date Created: Feb 13, 2021
"""

import sys
import string

def getTokens(line):
    '''
    Takes in each line of input and identifies tokens
    @params:line : line of the input file

    returns None
    '''
    tokens = line.split()
    l = len(tokens)

    if l==4 or (l>4 and tokens[4][0] =='#'):
        ### l =4 implies impedance element
        ### but incase a comment is present it may mislead the element type
        ### hence total tokens apart from comments should be 4
        element,n1,n2,value = tokens
        assert  n1.isalnum() and n2.isalnum(), "Node names need to be alphanumeric, please check Input file row {}".format(l+1)
        print(value,n2,n1,element)

    elif l == 6 or (l>6 and tokens[6][0] =='#'):
        ### l =6 implies VCVS/VCCS source
        ### but incase a comment is present it may mislead the element type
        ### hence total tokens apart from comments should be 6
        element,n1,n2,n3,n4,value = tokens
        assert n1.isalnum() and n2.isalnum() and n3.isalnum() and n4.isalnum(), "Node names need to be alphanumeric, please check Input file row {}".format(l+1)
        print(value,n4,n3,n2,n1,element)

    elif l==5 or (l>5 and tokens[5][0] =='#'):
        ### l =5 implies CCVS/CCCS source
        ### but incase a comment is present it may mislead the element type
        ### hence total tokens apart from comments should be 5
        element,n1,n2,V,value = tokens
        assert n1.isalnum() and n2.isalnum(),"Node names are alphanumeric, please check Input file row {}".format(l+1)
        print(value,V,n2,n1,element)

if __name__ =='__main__':

    assert len(sys.argv) == 2,'Please run this commad : python %s <inputfile>' % sys.argv[0]
    START = '.circuit'
    END = '.end'
    try:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
            print(lines)
            flag = 0
            contains = []
            for l in reversed(lines):
                tokens = l.split()
                if END == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#'):
                    flag = 1
                    continue
                if flag:
                    if START == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#'):
                        break
                    contains.append(l)
                    getTokens(l)
 
    except FileNotFoundError :
        print("File Not Found!!")