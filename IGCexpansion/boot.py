from __future__ import print_function
from CodonGeneconv import *
from copy import deepcopy
import os
from CodonGeneconFunc import *
import numpy as np
import random


#pip uninstall IGCexpansion
#pip install --user git+https://github.com/Tanchumin/Commonancestor.git



class bootstrap:
    def __init__(self,  tree_newick, alignment, paralog, model, post_dup='N1',
                 inibl=0.1, name=None,out=None):

        self.name_basic = alignment
        self.Model = model
        self.newicktree = tree_newick  # newick tree file loc
        self.paralog = paralog
        self.post_dup = post_dup
        self.nsites = None
        self.out=out

        self.tree = None  # store the tree dictionary used for json likelihood package parsing
        self.edge_to_blen = None
        self.edge_list = None  # kept all edges in the same order with x_rates
        self.node_to_num = None  # dictionary used for translating tree info from self.edge_to_blen to self.tree
        self.num_to_node = None

        self.name_to_seq = None

        self.inibl = inibl

        self.name = name

        bases = 'tcag'.upper()
        codons = [a + b + c for a in bases for b in bases for c in bases]
        amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'

        self.nt_to_state = {a: i for (i, a) in enumerate('ACGT')}
        self.state_to_nt = {i: a for (i, a) in enumerate('ACGT')}
        self.codon_table = dict(zip(codons, amino_acids))
        self.codon_nonstop = [a for a in self.codon_table.keys() if not self.codon_table[a] == '*']
        self.codon_to_state = {a.upper(): i for (i, a) in enumerate(self.codon_nonstop)}
        self.state_to_codon = {i: a.upper() for (i, a) in enumerate(self.codon_nonstop)}


        self.initialize_parameters()

    def initialize_parameters(self):
        self.get_tree()


    def get_tree(self):
        self.tree, self.edge_list, self.node_to_num ,length= read_newick(self.newicktree, self.post_dup)
        self.num_to_node = {self.node_to_num[i]:i for i in self.node_to_num}
        self.edge_to_blen = {edge:self.inibl for edge in self.edge_list}

    def nts_to_codons(self):
        for name in self.name_to_seq.keys():
            assert (len(self.name_to_seq[name]) % 3 == 0)
            tmp_seq = [self.name_to_seq[name][3 * j: 3 * j + 3] for j in range(int(len(self.name_to_seq[name]) / 3))]
            self.name_to_seq[name] = tmp_seq

    def separate_species_paralog_names(self, seq_name):
        assert (seq_name in self.name_to_seq)  # check if it is a valid sequence name
        matched_paralog = [paralog for paralog in self.paralog if paralog in seq_name]
        # check if there is exactly one paralog name in the sequence name
        return [seq_name.replace(matched_paralog[0], ''), matched_paralog[0]]



    def get_data(self,readname):

        seq_dict = SeqIO.to_dict(SeqIO.parse(readname, "fasta"))

        self.name_to_seq = {name: str(seq_dict[name].seq) for name in seq_dict.keys()}


        return self.name_to_seq




    def bootstrap(self):
        lst = np.linspace(1, 50, 50, endpoint=True)
        global namelist


        for boot in range(2):


            data_list=random.choices(lst, k=10)


            for index in range(10):
                i=int(data_list[index])
                readname = self.name_basic + str(i) + self.name
                if index ==0:
                    out = deepcopy(self.get_data(readname=readname))
                    namelist=list(out.keys())
                if index >=1:
                    for j in range(len(namelist)):
                         rowname=list(namelist)[j]
                         self.get_data(readname=readname)
                         out[rowname]=deepcopy(out[rowname]+self.name_to_seq[rowname])

            list_out=[]

            for i in range(len(namelist)):
                if i == 0:
                    p0 = ">" + list(self.name_to_seq.keys())[i] +  "\n"
                else:
                    p0 = "\n" + ">" + list(self.name_to_seq.keys())[i] +  "\n"


                p0 = p0 + out[list(self.name_to_seq.keys())[i]]
                list_out.append(p0)


            save_nameP =self.out+ str(boot+1)+ '/boot.fasta'
            with open(save_nameP, 'wb') as f:
                for file in list_out:
                    f.write(file.encode('utf-8'))


            out= None










if __name__ == '__main__':

    paralog = ['paralog0', 'paralog1']
    alignment_file = "/Users/txu7/Desktop/testfile/testy13t1/testy13t1/test copy "
    newicktree = '/Users/txu7/Desktop/testfile/testy13t1/testy13t1/test copy 1/YeastTree.newick'
    name="/MG94YNL069C_YIL133C_input_simu.fasta"
    out="/Users/txu7/Desktop/testfile/boot/test copy "

    model = "MG94"
    # Yixuan change deletec = False,  and do not change model="HKY"

    geneconv = bootstrap(newicktree, alignment_file, paralog, model=model,name=name,out=out)

    print(geneconv.bootstrap())
