#!/bin/python3

from __future__ import print_function
from IGCexpansion.em_pt import Embrachtau
from copy import deepcopy
import os

import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


if __name__ == '__main__':




    paralog = ['__Paralog1', '__Paralog2']
    alignment_file = './group_972_intron5_c.fasta'
    newicktree = './intronc.newick'
    name = "intron_972_5_c"
    #   name = 'tau99_01vss'
    #  Force ={0:np.exp(-0.71464127), 1:np.exp(-0.55541915), 2:np.exp(-0.68806275),3: np.exp( 0.74691342),4: np.exp( -0.5045814)}
    # %AG, % A, % C, kappa, tau
    # Force= {0:0.5,1:0.5,2:0.5,3:1,4:0}
    Force = None
    model = 'HKY'

    type = 'situation1'
    save_name = '../test/save/' + model + name + '_' + type + '_nonclock_save.txt'
    geneconv = Embrachtau(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name,kbound=5,kini=1.0,inibranch=0.01)


    geneconv.EM_branch_tau(K=1.2)

# print(joint_analysis.get_mle())
