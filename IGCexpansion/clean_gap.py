#Use to clean the gap in alignment data

from __future__ import print_function, absolute_import
from CodonGeneconFunc import *
import numpy as np
import os
import pickle

class Clean:
    def __init__(self,  tree_newick, alignment,paralog,model,post_dup = 'N1',inibl=0.1,deletec=True,name="sample"):

        self.seqloc      = alignment
        self.Model       = model
        self.newicktree  = tree_newick  # newick tree file loc
        self.paralog     = paralog
        self.post_dup = post_dup
        self.nsites = None
        self.deletec = deletec

        self.tree         = None        # store the tree dictionary used for json likelihood package parsing
        self.edge_to_blen = None
        self.edge_list    = None        # kept all edges in the same order with x_rates
        self.node_to_num  = None        # dictionary used for translating tree info from self.edge_to_blen to self.tree
        self.num_to_node  = None

        self.name_to_seq  =None

        self.inibl = inibl

        self.name=name

        bases = 'tcag'.upper()
        codons = [a+b+c for a in bases for b in bases for c in bases]
        amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'

        self.nt_to_state    = {a:i for (i, a) in enumerate('ACGT')}
        self.state_to_nt    = {i:a for (i, a) in enumerate('ACGT')}
        self.codon_table    = dict(zip(codons, amino_acids))
        self.codon_nonstop  = [a for a in self.codon_table.keys() if not self.codon_table[a]=='*']
        self.codon_to_state = {a.upper() : i for (i, a) in enumerate(self.codon_nonstop)}
        self.state_to_codon = {i : a.upper() for (i, a) in enumerate(self.codon_nonstop)}
        self.pair_to_state  = {pair:i for i, pair in enumerate(product(self.codon_nonstop, repeat = 2))}

    def get_tree(self):
        self.tree, self.edge_list, self.node_to_num = read_newick(self.newicktree, self.post_dup)
        self.num_to_node = {self.node_to_num[i]:i for i in self.node_to_num}
        self.edge_to_blen = {edge:self.inibl for edge in self.edge_list}

    def nts_to_codons(self):
        for name in self.name_to_seq.keys():
            assert(len(self.name_to_seq[name]) % 3 == 0)
            tmp_seq = [self.name_to_seq[name][3 * j : 3 * j + 3] for j in range(int(len(self.name_to_seq[name]) / 3) )]
            self.name_to_seq[name] = tmp_seq

    def separate_species_paralog_names(self, seq_name):
        assert(seq_name in self.name_to_seq)  # check if it is a valid sequence name
        matched_paralog = [paralog for paralog in self.paralog if paralog in seq_name]
        # check if there is exactly one paralog name in the sequence name
        return [seq_name.replace(matched_paralog[0], ''), matched_paralog[0]]

    def get_data(self,):

        self.get_tree()

        seq_dict = SeqIO.to_dict(SeqIO.parse(self.seqloc, "fasta"))
     #   print(seq_dict.keys())

        self.name_to_seq = {name: str(seq_dict[name].seq) for name in seq_dict.keys()}


        if self.Model == 'MG94':
            # Convert from nucleotide sequences to codon sequences.
            self.nts_to_codons()
            obs_to_state = deepcopy(self.codon_to_state)
            obs_to_state['---'] = -1
        else:
            obs_to_state = deepcopy(self.nt_to_state)
            obs_to_state['-'] = -1
            obs_to_state['?'] = -1



        if self.nsites is None:
            self.nsites = len(self.name_to_seq[list(self.name_to_seq.keys())[0]])
        else:
            for name in self.name_to_seq:
                self.name_to_seq[name] = self.name_to_seq[name][: self.nsites]

     #   print(list(self.name_to_seq.keys())[0])
        print ('number of sites to be analyzed: ', self.nsites)

        self.observable_names = [n for n in self.name_to_seq.keys() if
                                 self.separate_species_paralog_names(n)[0] in self.node_to_num.keys()]
        suffix_to_axis = {n: i for (i, n) in enumerate(list(set(self.paralog)))}
        self.observable_nodes = [self.node_to_num[self.separate_species_paralog_names(n)[0]] for n in
                                 self.observable_names]
        self.observable_axes = [suffix_to_axis[self.separate_species_paralog_names(s)[1]] for s in
                                self.observable_names]

        # Now convert alignment into state list
        iid_observations = []
    #    print(self.observable_names)
        # for i in range(self.nsites):
        #      print(i)
        #      print(self.name_to_seq["Human__Paralog1"][i])
        #

        dlsite=[]

        if self.deletec==False:

            for site in range(self.nsites):
                observations=[]
                for name in self.observable_names:
                    observation = obs_to_state[self.name_to_seq[name][site]]
                    observations.append(observation)
                    if ((self.name_to_seq[name][site]) == "-" or (self.name_to_seq[name][site]) == "?"):
                        dlsite.append(site)
                        print("delete")
                iid_observations.append(observations)

        else:

            for site in range(self.nsites):
                observations = []
                for name in self.observable_names:
                    if  name=="Chimp__Paralog1":
                        dd=1
                    elif name=="Chimp__Paralog2":
                        dd=1
                    else:
                       observation = obs_to_state[self.name_to_seq[name][site]]
                       observations.append(observation)
                       if ((self.name_to_seq[name][site]) == "-" or (self.name_to_seq[name][site]) == "?"):
                          dlsite.append(site)
                iid_observations.append(observations)

        #      print(iid_observations)
        dlsite=list(sorted(set(dlsite)))

    #    print(dlsite)
   #     print(len(dlsite))


        for dl in range(len(dlsite)):
            del iid_observations[dlsite[int(len(dlsite)-1-dl)]]

    #    print(iid_observations)
     #   print(self.observable_names)
        return iid_observations,int(len(dlsite))



    def change_into_seq(self):
        iid=self.get_data()

   #     print(self.state_to_nt)

        if not os.path.isdir('./prepared_input_clean/'):
            os.mkdir('./prepared_input_clean')

        seq_clean=[]
    #    print(iid[0])

        i=-1
        save_nameP = '../test/intron/' + self.name+'.fasta'
        with open(save_nameP, 'w+') as f:
      #      pickle.dump(seq_clean, f)
         if self.deletec==True:
            for  name in  self.observable_names:


                if name == "Chimp__Paralog1":
                    name =None
                elif name == "Chimp__Paralog2":
                    name = None
                else:
                    observations = ">"+name+"\n"
                    i=i+1


                    for site in range(self.nsites - iid[1]):
                     #   print(iid[0][site])
                        observations = observations+self.state_to_nt[iid[0][site][i]]
                      #  print(observations)

                    observations=observations+"\n"

                    f.write(observations)

         else:
             for name in self.observable_names:

                     observations = ">" + name + "\n"
                     i = i + 1

                     for site in range(self.nsites - iid[1] ):
                         #   print(iid[0][site])
                         observations = observations + self.state_to_nt[iid[0][site][i]]
                     #  print(observations)

                     observations = observations + "\n"

                     f.write(observations)






        # for site in range(self.nsites):
        #     observations = []
        #     for name in self.observable_names:
        #     #    print(name)
        #       #  print(obs_to_state[self.name_to_seq[name][site]])
        #         observation = obs_to_state[self.name_to_seq[name][site]]
        #         observations.append(observation)
        #     iid_observations.append(observations)
        # self.iid_observations = iid_observations







if __name__ == '__main__':


    paralog = ['__Paralog1', '__Paralog2']
    alignment_file = '../test/intron/testin.fasta'
    newicktree = '../test/intron/intron.newick'


    model="HKY"
    name="testin1"
# Yixuan change deletec = False,  and do not change model="HKY"

    geneconv = Clean(newicktree, alignment_file, paralog, model=model,name=name,deletec=True)
    geneconv.change_into_seq()