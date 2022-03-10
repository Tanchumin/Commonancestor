# coding=utf-8
# A separate file for Ancestral State Reconstruction
# Uses Alex Griffing's JsonCTMCTree package for likelihood and gradient calculation
# Xiang Ji
# xji4@tulane.edu
# Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function
from IGCexpansion.CodonGeneconv import ReCodonGeneconv
import argparse, os
from IGCexpansion.em_pt1 import *



def check_folder(folder_name):
    # if the folder doesn't exist,
    # this function creats it by assuming the outer folder exists (potential bug...)
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

def read_terminal_nodes(file_name):
    assert(os.path.isfile(file_name))  # make sure the file exists first
    with open(file, 'r') as f:
        terminal_node_list = [tip.strip() for tip in f.read().splitlines()]
    return terminal_node_list


if __name__ == '__main__':


    paralog = ['__Paralog2', '__Paralog1']
    alignment_file = './harpack2_combined.fasta'
    newicktree = './intronc.newick'
  #  Force ={0:np.exp(-0.71464127), 1:np.exp(-0.55541915), 2:np.exp(-0.68806275),3: np.exp( 0.74691342),4: np.exp( -0.5045814)}
    # %AG, % A, % C, kappa, tau
    #Force= {0:0.5,1:0.5,2:0.5,3:1,4:0}
    Force=None
    model = 'HKY'
    name = 'tau11111'
    type='situation1'

    save_folder = './save/'
    check_folder(save_folder)
    save_name = save_folder + model +  name+'_'+type+'_nonclock_save.txt'
    summary_folder = './summary/'
    check_folder(summary_folder)

    geneconv = Embrachtau1(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name,kbound=5,tauini=0.01,kini=2.1,inibranch=0.01)
    geneconv.EM_branch_tau(K=-1.8)




