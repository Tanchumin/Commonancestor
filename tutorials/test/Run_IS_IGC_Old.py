#! /usr/bin/python3
# A separate file for Ancestral State Reconstruction
# Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function
from IGCexpansion.CodonGeneconv import *
from IGCexpansion.em_pt1 import *
from IGCexpansion.gls_seq import *



if __name__ == '__main__':


    name = "YBL087C_YER117W_input"

    paralog = ['YBL087C', 'YER117W']
    alignment_file = './' + name + '.fasta'
    newicktree = './YeastTree.newick'



    save_path='./'

    Force = None
    model = 'MG94'
    save_name = model + name


    geneconv = Embrachtau1(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                                 save_path=save_path, save_name=save_name)

    save_name_simu = model + name+"_simu"


    self = GSseq(geneconv=geneconv, sizen=400, ifmakeQ=False,Model=model,save_path=save_path, save_name=save_name_simu)

    aaa = self.topo()
    self.trans_into_seq(ini=aaa[0], name_list=aaa[1])

    simulate_file = save_path + save_name_simu + ".fasta"
    paralog_simu = ['paralog0', 'paralog1']
    save_name1 = save_path + save_name+"_simu"

    geneconv_simu = Embrachtau1(newicktree, simulate_file, paralog_simu, Model=model, Force=Force, clock=None,
                                save_path=save_path, save_name=save_name1)


    geneconv_simu.sum_branch(MAX=5,K=1.5)








