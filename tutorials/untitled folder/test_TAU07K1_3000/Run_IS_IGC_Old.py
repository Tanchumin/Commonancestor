#! /usr/bin/python3
# A separate file for Ancestral State Reconstruction
# Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function
import jsonctmctree.ll, jsonctmctree.interface
from IGCexpansion.CodonGeneconv import *
from IGCexpansion.acR import *
from IGCexpansion.em_pt import *
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from numpy import random
from scipy import linalg
import copy
from scipy.stats import poisson

from IGCexpansion.CodonGeneconFunc import isNonsynonymous
import pickle
import json
import numpy.core.multiarray



if __name__ == '__main__':


    name = "si"

    paralog = ['paralog0', 'paralog1']
    alignment_file = './' + name + '.fasta'
    newicktree = './sample1.newick'

    #   name = 'tau99_01vss'
    #  Force ={0:np.exp(-0.71464127), 1:np.exp(-0.55541915), 2:np.exp(-0.68806275),3: np.exp( 0.74691342),4: np.exp( -0.5045814)}
    # %AG, % A, % C, kappa, tau
    # Force= {0:0.5,1:0.5,2:0.5,3:1,4:0}
    Force = None
    model = 'HKY'

    type = 'situation1'
    save_name = model+name
    geneconv = Embrachtau(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                          save_path='../test/save/', save_name=save_name,noboundk=True)
                          
    geneconv.EM_branch_tau(epis=0.005,MAX=8,K=2)



# [0.97643333 0.         0.88303333 0.87493333 0.76533333 0.77053333
#  0.68076667 0.68306667 0.6208     0.62013333]