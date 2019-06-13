import CodonGeneconFunc
import CodonGeneconv
from Bio import  SeqIO
from IGCexpansion.CodonGeneconFunc import *
import argparse
#from jsonctmctree.extras import optimize_em
import ast


paralog = ['ECP', 'EDN']
alignment_file = '../test/EDN_ECP_Cleaned.fasta'
newicktree = '../test/input_tree.newick'
Force1 = None
model = 'MG94'
seq_dict = SeqIO.to_dict(SeqIO.parse( alignment_file, "fasta" ))
name_to_seq = {name:str(seq_dict[name].seq) for name in seq_dict.keys()}

bases = 'tcag'.upper()
codons = [a + b + c for a in bases for b in bases for c in bases]
amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
codon_table    = dict(zip(codons, amino_acids))
codon_nonstop  = [a for a in codon_table.keys() if not codon_table[a]=='*']
codon_to_state = {a.upper() : i for (i, a) in enumerate(codon_nonstop)}
state_to_codon = {i : a.upper() for (i, a) in enumerate(codon_nonstop)}
pair_to_state  = {pair:i for i, pair in enumerate(product(codon_nonstop, repeat = 2))}


def nts_to_codons():
    for name in name_to_seq.keys():
        assert (len(name_to_seq[name]) % 3 == 0)
        tmp_seq = [name_to_seq[name][3 * j: 3 * j + 3] for j in range(len(name_to_seq[name]) / 3)]
        name_to_seq[name] = tmp_seq


nts_to_codons()
obs_to_state = codon_to_state
obs_to_state['---'] = -1
print(codon_nonstop)


def get_MG94Basic():
    Qbasic = np.zeros((61, 61), dtype=float)
    for ca in codon_nonstop:
        for cb in codon_nonstop:
            if ca == cb:
                continue
            Qbasic[codon_to_state[ca], codon_to_state[cb]] = get_MG94BasicRate(ca, cb, self.pi, self.kappa,
                                                                                         self.omega, self.codon_table)
    expected_rate = np.dot(self.prior_distribution, Qbasic.sum(axis=1))
    Qbasic = Qbasic / expected_rate
    return Qbasic

for i, pair in enumerate(product(codon_nonstop, repeat=2)):
    # use ca, cb, cc to denote codon_a, codon_b, codon_c, where cc != ca, cc != cb
    ca, cb = pair
    sa = codon_to_state[ca]
    sb = codon_to_state[cb]
    for cc in codon_nonstop:
        if cc == ca or cc == cb:
            continue
        sc = codon_to_state[cc]



