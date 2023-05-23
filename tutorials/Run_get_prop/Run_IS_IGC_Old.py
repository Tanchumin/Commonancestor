#! /usr/bin/python3
# A separate file for Ancestral State Reconstruction
# Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function
from IGCexpansion.CodonGeneconv import *
from IGCexpansion.em_pt1 import *
from IGCexpansion.gls_seq import *
import re

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]



if __name__ == '__main__':

    files = os.listdir('./' )
    files = ['./'  + file for file in files if 'fasta' in file]
 #   print(files[0])
    paralog_list = [file.replace('_input.fasta', '') for file in files]
    paralog_list = [file.replace('./', '') for file in paralog_list]
    paralog_list.sort(key=natural_keys)
    paralog_list = [file.split("_") for file in paralog_list]
#    print(paralog_list[0])


    paralog = paralog_list[0]
    alignment_file = files[0]
    name=paralog[0]+"_"+paralog[1]+"_input"

    newicktree = './YeastTree.newick'
    Force = None
    model = 'MG94'

    save_path='./'

    type = 'situation_new'
    save_name = model + name
    geneconv = Embrachtau1(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                                  save_path=save_path, save_name=save_name)


    save_name_simu = model + name + "_simu"
    len_seq=geneconv.nsites


    self = GSseq(geneconv=geneconv, sizen=len_seq, ifmakeQ=False, Model=model, save_path=save_path,
                     save_name=save_name_simu, ifDNA=True)

    self.proportion_change_IGC(repeats=10)
   
