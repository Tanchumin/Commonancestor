import CodonGeneconFunc
import CodonGeneconv
from Bio import  SeqIO
from IGCexpansion.CodonGeneconFunc import *
import argparse
#from jsonctmctree.extras import optimize_em
import ast

from CodonGeneconv import ReCodonGeneconv
import argparse, os
import numpy as np

def check_folder(folder_name):
    # if the folder doesn't exist,
    # this function creats it by assuming the outer folder exists (potential bug...)
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

if __name__ == '__main__':

    paralog = ['EDN', 'ECP']
    alignment_file = '../test/EEEE.fasta'
    newicktree = '../test/EEEE.newick'
    Force = {5:0,6:0,7:1}
    model = 'MG94'


    name='EDN_ECP'
    type='situation1'

    save_name = '../test/save/' + model + name+'_'+type+'_nonclock_save.txt'
    test = ReCodonGeneconv(newicktree, alignment_file, paralog, ptau1=1, ptau2=1, pc=1, eqtau12=True ,Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name)

    test.get_mle()




