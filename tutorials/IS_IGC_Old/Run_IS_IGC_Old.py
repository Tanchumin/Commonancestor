# coding=utf-8
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


    name = "YBL087C_YER117W_input"

    paralog = ['YBL087C', 'YER117W']
    alignment_file = './' + name + '.fasta'
    newicktree = './YeastTree.newick'

    #   name = 'tau99_01vss'
    #  Force ={0:np.exp(-0.71464127), 1:np.exp(-0.55541915), 2:np.exp(-0.68806275),3: np.exp( 0.74691342),4: np.exp( -0.5045814)}
    # %AG, % A, % C, kappa, tau
    # Force= {0:0.5,1:0.5,2:0.5,3:1,4:0}
    Force = None
    model = 'HKY'

    type = 'situation1'
    save_name = model+name
 #   geneconv = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
  #                             save_path='./', save_name=save_name)
#
  #  self = AncestralState1(geneconv)

    geneconv = Embrachtau(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                          save_path='../test/save/', save_name=save_name)


   # geneconv.EM_branch_tau()
    for name in geneconv.name_to_seq.keys():
        print(name)

    print(geneconv.tree['col'])
    print(geneconv.tree['row'])
    print(geneconv.observable_nodes)
    print(geneconv.observable_axes)



 #   self.get_paralog_diverge()
