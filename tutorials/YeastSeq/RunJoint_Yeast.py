#!/bin/python3

from __future__ import print_function
import jsonctmctree.ll, jsonctmctree.interface
from IGCexpansion.CodonGeneconv import *
from IGCexpansion.em_pt import *
from IGCexpansion.joint_ana import *
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from numpy import random
from scipy import linalg
import copy

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

    inputFolder = 'YeastSeq'
    outputName = 'JointOmega_Yeast'

    # parameters
    Force = None
    IGC_Omega = 0.8
    Tau_Omega = None
    Model = 'MG94'

    files = os.listdir('../' + inputFolder)
    files = ['../'+inputFolder+'/' + file for file in files if 'fasta' in file]
    files.sort(key=natural_keys)
    paralog_list = [file.replace('_input.fasta', '') for file in files]
    paralog_list = [file.replace('../YeastSeq/', '') for file in paralog_list]
    paralog_list.sort(key=natural_keys)
    paralog_list = [file.split("_") for file in paralog_list]
    # print(paralog_list)
    # print(files)
    #paralog_list = [['01_'+re.findall(r'\d+', file)[0], '02_'+re.findall(r'\d+', file)[0]] for file in files]
    alignment_file_list = files
    newicktree = '../'+inputFolder+'/YeastTree.newick'

    Shared = [5]


    save_path = '../JointAnalysisResult/' + outputName + '/save/'
    summary_path = '../JointAnalysisResult/' + outputName + '/summary/'
    os.makedirs(save_path, exist_ok=True) # save parameters
    os.makedirs(summary_path, exist_ok=True) # save summary
    print('start to analyze')
    print('Input: ' + inputFolder)
    print('Job name: ' + outputName)
    print('IGC_Omega: ' + str(IGC_Omega))
    print('Tau_Omega: ' + str(Tau_Omega))
    print('number of files: ' + str(len(files)))
    
    joint_analysis = JointAnalysis(alignment_file_list,  newicktree, paralog_list, Shared = Shared,
                                   Model = Model, Force = Force,
                                   save_path = './save/',inibranch=0.1, kini=1.4, tauini=2.004)
                                   
    print(joint_analysis.em_joint())
