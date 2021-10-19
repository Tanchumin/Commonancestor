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

    
    paralog = ['paralog0', 'paralog1']
    alignment_file = './si.fasta'
    newicktree = './sample1.newick'
    name = 'k2tau4t5'

    #   name = 'tau99_01vss'
    #  Force ={0:np.exp(-0.71464127), 1:np.exp(-0.55541915), 2:np.exp(-0.68806275),3: np.exp( 0.74691342),4: np.exp( -0.5045814)}
    # %AG, % A, % C, kappa, tau
    # Force= {0:0.5,1:0.5,2:0.5,3:1,4:0}
    Force = None
    model = 'MG94'

    type = 'situation1'
    save_name = model+name
    geneconv = Embrachtau(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                          save_path='../test/save/', save_name=save_name)
                          
    geneconv.EM_branch_tau(force= {6:0.01},K=0.01)


