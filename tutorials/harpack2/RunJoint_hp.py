#!/bin/python3

from __future__ import print_function
from IGCexpansion.joint_ana import *
from copy import deepcopy
import os

import re

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

if __name__ == '__main__':

    inputFolder = 'harpack2'
    outputName = 'JointOmega_Yeast'

    # parameters
    Force = None
    IGC_Omega = 0.8
    Tau_Omega = None
    Model = 'HKY'

    files = os.listdir('../' + inputFolder)
    files = ['../'+inputFolder+'/' + file for file in files if 'fasta' in file]
    files.sort(key=natural_keys)
    paralog_list = [file.replace('_c.fasta', '') for file in files]
    paralog_list = [file.replace('../harpack2/', '') for file in paralog_list]
    paralog_list.sort(key=natural_keys)
    paralog_list = [file.split("_") for file in paralog_list]
    # print(paralog_list)
    # print(files)
    #paralog_list = [['01_'+re.findall(r'\d+', file)[0], '02_'+re.findall(r'\d+', file)[0]] for file in files]
    alignment_file_list = files
    newicktree = '../'+inputFolder+'/intronc.newick'

    shared_parameters_for_k = [4]


    save_path = './' + outputName + '/save/'
    summary_path = './' + outputName + '/summary/'
    os.makedirs(save_path, exist_ok=True) # save parameters
    os.makedirs(summary_path, exist_ok=True) # save summary
    print('start to analyze')
    print('Input: ' + inputFolder)
#    print('Job name: ' + outputName)

    print('number of files: ' + str(len(files)))

    print(alignment_file_list)
    dd=['__Paralog1', '__Paralog2']
    paralog_list=[]
    for i in range(len(alignment_file_list)):
        paralog_list.append(dd)

  #  print(paralog_list)
    
   # joint_analysis =JointAnalysis_nest(alignment_file_list,  newicktree, paralog_list, Shared = Shared,
      #                             IGC_Omega = None, Model = Model, Force = Force,Force_share={4:0},
   #                                    shared_parameters_for_k=[4, 5], Force_share_k={4: 0, 5: 0},
     #                                  tauini=1.2,inibranch=0.2,kini=0.1,
     #                              save_path = './save/')
    joint_analysis = JointAnalysis(alignment_file_list, newicktree, paralog_list, Shared=[],
                                            IGC_Omega = None, Model = Model,
                                            shared_parameters_for_k=[5],
                                            kini=0.5,
                                            save_path = './save/')
 #0.2 0.4 best ini by now 28965.8

  #  print(joint_analysis.get_seq_mle())
    print(joint_analysis.em_joint())

   # print(joint_analysis.get_mle())
