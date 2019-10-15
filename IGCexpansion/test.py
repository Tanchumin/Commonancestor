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
    Force = {7:1}
    model = 'MG94'


    name='YYYYYY'
    type='situation8'

    save_name = '../test/save/' + model+'_' + name+'_'+type+'_nonclock_save.txt'

    summary_folder = '../summary/'
    check_folder(summary_folder)

    test = ReCodonGeneconv(newicktree, alignment_file, paralog, ptau1=1, ptau2=1, pc=1, eqtau12=True ,Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name)

    test.get_mle()
    test.get_ExpectedNumGeneconv()
    test.get_individual_summary(summary_path=summary_folder,name=name,type=type)

    IGC_sitewise_lnL_file = summary_folder + model + '_'+name+"_"+type+'_'+'_'.join(paralog) + '_nonclock_sitewise_lnL_summary.txt'
    test.get_sitewise_loglikelihood_summary(IGC_sitewise_lnL_file)

    test.get_SitewisePosteriorSummary(summary_path=summary_folder,name=name,type=type)





