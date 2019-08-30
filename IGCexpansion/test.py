import CodonGeneconFunc
import CodonGeneconv
from Bio import  SeqIO
from IGCexpansion.CodonGeneconFunc import *
import argparse
#from jsonctmctree.extras import optimize_em
import ast

from IGCexpansion.CodonGeneconv import ReCodonGeneconv
import argparse, os
import numpy as np

if __name__ == '__main__':

    paralog = ['EDN', 'ECP']
    alignment_file = '../test/EEEE.fasta'
    newicktree = '../test/EEEE.newick'
    Force = {5:0,6:0,7:1}
    model = 'MG94'


    name='EDN_ECP'
    type='situation7'
    save_name = '../test/save/' + model + name+'_'+type+'_nonclock_save.txt'
    test = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name,ptau1=1,ptau2=1,pc=1,eqtau12=True)

    test.get_mle()




