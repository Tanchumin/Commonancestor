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
import re

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]


if __name__ == '__main__':

  #  inputFolder = 'YeastSeq'

    files = os.listdir('./' )
    files = ['./'  + file for file in files if 'fasta' in file]
 #   print(files[0])
    paralog_list = [file.replace('_input.fasta', '') for file in files]
    paralog_list = [file.replace('./', '') for file in paralog_list]
    paralog_list.sort(key=natural_keys)
    paralog_list = [file.split("_") for file in paralog_list]




    paralog = paralog_list[0]
    alignment_file = files[0]

    newicktree = './YeastTree.newick'

    Force = None
    model = 'MG94'

    type = 'situation1'
    save_name = model


    geneconv = Embrachtau(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                          save_path='../test/save/', save_name=save_name)



    geneconv.EM_branch_tau(ifdnalevel=False)




 #   self.get_paralog_diverge()
