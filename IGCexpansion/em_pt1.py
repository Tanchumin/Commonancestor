
# coding=utf-8
# A separate file for Ancestral State Reconstruction
#EM for branch tau
# Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function, absolute_import
import jsonctmctree.ll, jsonctmctree.interface
from CodonGeneconv import *
from copy import deepcopy
import os
import numpy as np
from scipy import linalg
from IGCexpansion.CodonGeneconFunc import isNonsynonymous
import numdifftools as nd





### iniset is used for joint ana seq version
class Embrachtau1:
    def __init__(self, tree_newick, alignment, paralog, Model='MG94', IGC_Omega=None, Tau_Omega = None, nnsites=None, clock=False,joint=False,
                 Force=None, save_path='./save/', save_name=None, post_dup='N1',kbound=5.1,ifmodel="old",inibranch=0.1,noboundk=True,
                 kini=4.1,tauini=1,omegaini=0.5,dwell_id=True,ifDNA=True,if_rerun=True):

        self.newicktree = tree_newick  # newick tree file loc
        self.seqloc = alignment  # multiple sequence alignment, now need to remove gap before-hand
        self.paralog = paralog  # parlaog list
        self.nsites = nnsites  # number of sites in the alignment used for calculation
        self.Model = Model
        self.IGC = ''  # indicates if or not IGC in ancestral compare result
        self.ll = 0.0  # Store current log-likelihood
        self.Force = Force  # parameter constraints only works on self.x not on x_clock which should be translated into self.x first
        self.clock = clock  # molecular clock control
        self.post_dup = post_dup  # store first post-duplication node name
        self.save_path = save_path  # location for auto-save files
        self.save_name = save_name  # save file name
        self.auto_save = 0  # auto save control
        self.IGC_Omega = IGC_Omega  # separate omega parameter for IGC-related nonsynonymous changes

        self.logzero = -15.0  # used to avoid log(0), replace log(0) with -15
        self.infinity = 1e6  # used to avoid -inf in gradiance calculation of the clock case
        self.minlogblen = -9.0  # log value, used for bond constraint on branch length estimates in get_mle() function

        # Tree topology related variable
        self.tree = None  # store the tree dictionary used for json likelihood package parsing
        self.edge_to_blen = None  # dictionary store the unpacked tree branch length information {(node_from, node_to):blen}
        self.edge_list = None  # kept all edges in the same order with x_rates
        self.node_to_num = None  # dictionary used for translating tree info from self.edge_to_blen to self.tree
        self.num_to_node = None  # dictionary used for translating tree info from self.tree to self.edge_to_blen

        # Constants for Sequence operations
        bases = 'tcag'.upper()
        codons = [a + b + c for a in bases for b in bases for c in bases]
        amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'

        self.nt_to_state = {a: i for (i, a) in enumerate('ACGT')}
        self.state_to_nt = {i: a for (i, a) in enumerate('ACGT')}
        self.codon_table = dict(zip(codons, amino_acids))
        self.codon_nonstop = [a for a in self.codon_table.keys() if not self.codon_table[a] == '*']
        self.codon_to_state = {a.upper(): i for (i, a) in enumerate(self.codon_nonstop)}
        self.state_to_codon = {i: a.upper() for (i, a) in enumerate(self.codon_nonstop)}
        self.pair_to_state = {pair: i for i, pair in enumerate(product(self.codon_nonstop, repeat=2))}

        # Tip data related variable
        self.name_to_seq = None  # dictionary store sequences
        self.observable_names = None  # list of extent species + paralog name ( = self.name_to_seq.keys())
        self.observable_nodes = None  # list of extent species numbers (node_to_num)
        self.observable_axes = None  # list of paralog numbers
        self.iid_observations = None  # list of multivariate states

        # Rate matrix related variable
        self.x_process = None  # values of process parameters (untransformed, log, or exp(-x))
        self.x_rates = None  # values of blen (untransformed, log, or exp(-x))
        self.x = None  # x_process + x_rates
        self.x_Lr = None  # values of clock blen parameters
        self.x_clock = None  # x_process + Lr
        self.pi = None  # real values
        self.kappa = 0.87  # real values
        self.omega = omegaini  # real values

        # set ini value for key parameters
        self.tau = tauini  # initial tau
     #   self.tau_F=tauini # initial tau
    #    self.k_F = kini # initial K
        self.K=kini # initial K
        self.inibranch=inibranch # initial branch
        self.sites=None # store internal nodes from ancestral inference
        self.processes = None # list of basic and geneconv rate matrices. Each matrix is a dictionary used for json parsing

        self.ifmodel=ifmodel # "old" model without studying K,
        self.id=None # the paralog identity level for all branches
        self.bound=False # set bound for other paramters in mle
        self.kbound=kbound # set bound for K in mle
        self.noboundk=noboundk  # noboundk  from -inf to inf

        self.hessian = False
        #debug for shared version
        self.edge_de=None

        self.save_name1 = save_name+"K"
        self.auto_save1=0

        self.scene_ll = None  # used for lnL calculation

        # Prior distribution on the root
        self.prior_feasible_states = None
        self.prior_distribution = None

        # ancestral reconstruction series
        self.reconstruction_series = None  # nodes * paralogs * 'string'

        self.joint=joint  # Initialize all parameters to construct internal paralog

        self.index=0
        self.ifexp=False
        self.ifDNA=ifDNA


        #### for bs tau
        self.Q=None
        self.dic_col=None
        self.branch_tau=2.1
        self.ini=None
        self.end=None
        self.time=0.1
        self.if_rerun=if_rerun

        self.dwell_id=dwell_id

        self.only_hessian = False
        self.initialize_parameters()




    def initialize_parameters(self):
        self.get_tree()
        self.get_data()
        self.get_initial_x_process()

        save_file = self.get_save_file_name()

        if self.joint==False:

            if os.path.isfile(save_file):  # if the save txt file exists and not empty, then read in parameter values
                if os.stat(save_file).st_size > 0:
                    self.initialize_by_save(save_file)
                    print('Successfully loaded parameter value from ' + save_file)



    def get_tree(self):
        self.tree, self.edge_list, self.node_to_num,length = read_newick(self.newicktree, self.post_dup)
        self.num_to_node = {self.node_to_num[i]: i for i in self.node_to_num}
        self.edge_to_blen={}
        for i in range(len(length)):
            self.edge_to_blen[self.edge_list[i]] = length[i]



    def nts_to_codons(self):
        for name in self.name_to_seq.keys():
            assert (len(self.name_to_seq[name]) % 3 == 0)
            tmp_seq = [self.name_to_seq[name][3 * j: 3 * j + 3] for j in range(int(len(self.name_to_seq[name]) / 3))]
            self.name_to_seq[name] = tmp_seq

    def get_data(self):
        seq_dict = SeqIO.to_dict(SeqIO.parse(self.seqloc, "fasta"))
        self.name_to_seq = {name: str(seq_dict[name].seq) for name in seq_dict.keys()}

        if self.Model == 'MG94':
            # Convert from nucleotide sequences to codon sequences.
            self.nts_to_codons()
            obs_to_state = deepcopy(self.codon_to_state)
            obs_to_state['---'] = -1
        else:
            obs_to_state = deepcopy(self.nt_to_state)
            obs_to_state['-'] = -1

        # change the number of sites for calculation if requested
        if self.nsites is None:
            self.nsites = len(self.name_to_seq[list(self.name_to_seq.keys())[0]])
        else:
            for name in self.name_to_seq:
                self.name_to_seq[name] = self.name_to_seq[name][: self.nsites]
        print('number of sites to be analyzed: ', self.nsites)

        # assign observable parameters
        self.observable_names = [n for n in self.name_to_seq.keys() if
                                 self.separate_species_paralog_names(n)[0] in self.node_to_num.keys()]
        suffix_to_axis = {n: i for (i, n) in enumerate(list(set(self.paralog)))}
        self.observable_nodes = [self.node_to_num[self.separate_species_paralog_names(n)[0]] for n in
                                 self.observable_names]
        self.observable_axes = [suffix_to_axis[self.separate_species_paralog_names(s)[1]] for s in
                                self.observable_names]

        # Now convert alignment into state list
        iid_observations = []
        for site in range(self.nsites):
            observations = []
            for name in self.observable_names:
                observation = obs_to_state[self.name_to_seq[name][site]]
                observations.append(observation)
            iid_observations.append(observations)
        self.iid_observations = iid_observations

    def separate_species_paralog_names(self, seq_name):
        assert (seq_name in self.name_to_seq)  # check if it is a valid sequence name
        matched_paralog = [paralog for paralog in self.paralog if paralog in seq_name]

        # check if there is exactly one paralog name in the sequence name
        return [seq_name.replace(matched_paralog[0], ''), matched_paralog[0]]

    def get_initial_x_process(self, transformation='log'):


        count = np.array([0, 0, 0, 0], dtype=float)  # count for A, C, G, T in all seq
        for name in self.name_to_seq:
            for i in range(4):
                count[i] += ''.join(self.name_to_seq[name]).count('ACGT'[i])
        count = count / count.sum()


        if self.ifmodel=="old":

            self.x_rates = np.log(np.array([ self.edge_to_blen[edge] for edge in self.edge_to_blen.keys()]))
            if self.Model == 'MG94':
                # x_process[] = %AG, %A, %C, kappa, omega, tau

                if self.IGC_Omega is None:
                    self.x_process = np.log(
                        np.array([count[0] + count[2], count[0] / (count[0] + count[2]), count[1] / (count[1] + count[3]),
                                  self.kappa, self.omega, self.tau]))
                else:
                    self.x_process = np.log(
                        np.array([count[0] + count[2], count[0] / (count[0] + count[2]), count[1] / (count[1] + count[3]),
                                  self.kappa, self.omega, self.IGC_Omega, self.tau]))

            elif self.Model == 'HKY':
                # x_process[] = %AG, %A, %C, kappa, tau
                self.omega = 1.0
                self.x_process = np.log(
                    np.array([count[0] + count[2], count[0] / (count[0] + count[2]), count[1] / (count[1] + count[3]),
                              self.kappa, self.tau]))
        else:
            save_file = self.get_save_file_name()


            if self.Model == 'MG94':

                if self.noboundk==False:
                    if self.IGC_Omega is None:
                        self.x_process = np.log(
                            np.array([np.exp(self.x_process[0]),np.exp(self.x_process[1]), np.exp(self.x_process[2]),
                                      self.kappa, self.omega, self.tau,self.K]))
                    else:
                        self.x_process = np.log(
                            np.array([np.exp(self.x_process[0]), np.exp(self.x_process[1]), np.exp(self.x_process[2]),
                                  self.kappa, self.omega, self.IGC_Omega, self.tau,self.K]))

                else:
                    if self.IGC_Omega is None:
                        self.x_process = np.append(np.log(
                            np.array([np.exp(self.x_process[0]), np.exp(self.x_process[1]), np.exp(self.x_process[2]),
                                      self.kappa, self.omega, self.tau])), self.K)
                    else:
                        self.x_process = np.append(np.log(
                            np.array([np.exp(self.x_process[0]), np.exp(self.x_process[1]), np.exp(self.x_process[2]),
                                      self.kappa, self.omega, self.IGC_Omega, self.tau])), self.K)

            else:
                # x_process[] = %AG, %A, %C, kappa, tau
                self.omega = 1.0
                if self.noboundk == False:
                    self.x_process = np.log(
                        np.array([np.exp(self.x_process[0]), np.exp(self.x_process[1]), np.exp(self.x_process[2]),
                                  self.kappa, self.tau,self.K]))
                else:
                    self.x_process = np.append(np.log(
                        np.array([np.exp(self.x_process[0]), np.exp(self.x_process[1]), np.exp(self.x_process[2]),
                                  self.kappa, self.tau])), self.K)

            if self.joint == False:
                if os.path.isfile(save_file):  # if the save txt file exists and not empty, then read in parameter values
                    if os.stat(save_file).st_size > 0:
                                self.initialize_by_save(save_file)
                                print('Successfully loaded parameter value from ' + save_file)
                                #                        self.if_rerun=False

                                if self.Model == "MG94":
                                    self.k = self.x[6]
                                else:
                                    self.k = self.x[5]

                                # EM update for id
                                self.update_by_x(self.x)
                                self.id = self.compute_paralog_id()
                                print("dddddd111111111")
                                self.update_by_x(self.x)
                                self.id = self.compute_paralog_id()
                                self.update_by_x(self.x)
                                self.id = self.compute_paralog_id()
                                self.update_by_x(self.x)


        if transformation == 'log' :
            self.x = np.concatenate((self.x_process, self.x_rates))
        elif transformation == 'log' and self.ifmodel=="EM_reduce":
            self.x = np.concatenate(self.x_process)

        elif transformation == 'None':
            self.x_process = np.exp(self.x_process)
            self.x_rates = np.exp(self.x_rates)
        elif transformation == 'Exp_Neg':
            self.x_process = np.exp(-np.exp(self.x_process))
            self.x_rates = np.exp(-np.exp(self.x_rates))
        self.x = np.concatenate((self.x_process, self.x_rates))

        if self.clock:  # set-up x_clock if it's a clock model
            l = len(self.edge_to_blen) / 2 + 1  # number of leaves
            self.x_Lr = np.log(np.ones(int(l)) * 0.6)

            if transformation == 'log':
                self.x_clock = np.concatenate((self.x_process, self.x_Lr))
            elif transformation == 'None':
                self.x_Lr = np.exp(self.x_Lr)
            elif transformation == 'Exp_Neg':
                self.x_Lr = np.exp(-np.exp(self.x_Lr))
            self.x_clock = np.concatenate((self.x_process, self.x_Lr))
            self.unpack_x_clock(transformation=transformation)

        self.update_by_x(transformation=transformation)


    # for K
    def ini_by_file(self):
        save_file1 = self.get_save_file_name()

        if os.path.isfile(save_file1):  # if the save txt file exists and not empty, then read in parameter values
            if os.stat(save_file1).st_size > 0:
                self.update_by_x(self.x)
                self.id = self.compute_paralog_id()
                self.update_by_x(self.x)
                self.id = self.compute_paralog_id()
                self.update_by_x(self.x)
                self.id = self.compute_paralog_id()
                self.update_by_x(self.x)
                print(self.id)
                print('Successfully loaded parameter value from ' + save_file1)

    def update_by_x_clock(self, x_clock=None, transformation='log'):
        if not x_clock is None:
            self.x_clock = x_clock
        self.unpack_x_clock(transformation=transformation)
        self.update_by_x(transformation=transformation)

    def unpack_x_clock(self, transformation):
        assert (self.clock)
        nEdge = len(self.edge_to_blen)  # number of edges
        assert (nEdge % 2 == 0)
        l = int(nEdge / 2) + 1  # number of leaves
        k = l - 1  # number of internal nodes. The notation here is inconsistent with Alex's for trying to match my notes.
        if transformation == 'log':
            self.x_process, self.x_Lr = self.x_clock[:-l], np.exp(self.x_clock[-l:])
        elif transformation == 'None':
            self.x_process, self.x_Lr = self.x_clock[:-l], self.x_clock[-l:]
        elif transformation == 'Exp_Neg':
            self.x_process, self.x_Lr = self.x_clock[:-l], self.x_clock[-l:]
            self.x_clock[0] = - np.log(self.x_clock[0])

        # Now update self.x by using self.x_clock
        leaf_branch = [edge for edge in self.edge_to_blen.keys() if
                       edge[0][0] == 'N' and str.isdigit(edge[0][1:]) and not str.isdigit(edge[1][1:])]
        out_group_branch = [edge for edge in leaf_branch if edge[0] == 'N0' and not str.isdigit(edge[1][1:])][0]
        internal_branch = [x for x in self.edge_to_blen.keys() if not x in leaf_branch]
        assert (len(
            internal_branch) == k - 1)  # check if number of internal branch is one less than number of internal nodes

        leaf_branch.sort(
            key=lambda node: int(node[0][1:]))  # sort the list by the first node number in increasing order
        internal_branch.sort(
            key=lambda node: int(node[0][1:]))  # sort the list by the first node number in increasing order

        # Now update blen with fixed order:
        # Always start from the root and internal-tip branch first
        for i in range(len(internal_branch)):
            edge = internal_branch[i]
            self.x_rates[2 * i] = self.blen_from_clock(edge)
            edge = leaf_branch[i]
            self.x_rates[2 * i + 1] = self.blen_from_clock(edge)
        for j in range(len(leaf_branch[i + 1:])):
            edge = leaf_branch[i + 1 + j]
            self.x_rates[- len(leaf_branch[i + 1:]) + j] = self.blen_from_clock(edge)
        # update self.x so that all parameters can be updated by update_by_x
        if transformation == 'log':
            self.x_rates = np.array([np.log(rate) if rate > 0 else self.logzero for rate in self.x_rates])
            self.x = np.concatenate((self.x_process, self.x_rates))
        elif transformation == 'None':
            self.x = np.concatenate((self.x_process, self.x_rates))
        elif transformation == 'Exp_Neg':
            self.x = np.concatenate((self.x_process, np.exp(-self.x_rates)))

    def blen_from_clock(self, edge):
        assert (edge in self.edge_to_blen.keys())
        if edge[0] == 'N0':
            if str.isdigit(edge[1][1:]):  # (N0, N1) branch
                return self.x_Lr[0] * self.x_Lr[1] * (1 - self.x_Lr[2])
            else:
                return self.x_Lr[0] * (2 - self.x_Lr[1])

        else:
            tmp_k = int(edge[0][1:])
            if str.isdigit(edge[1][1:]):  # ( N_temp_k, N_temp_k+1 ) branch
                return reduce(mul, self.x_Lr[: (tmp_k + 2)], 1) * (1 - self.x_Lr[tmp_k + 2])
            else:  # ( N_temp_k, leaf ) branch
                return reduce(mul, self.x_Lr[: (tmp_k + 2)], 1)

    def update_by_x(self, x=None, transformation='log'):
        k = len(self.edge_to_blen)

        if x is not None:
            self.x = x
        self.x_process, self.x_rates = self.x[:-k], self.x[-k:]

        Force_process = None
        Force_rates = None

        if self.Force != None:
      #      print(self.Force)
            Force_process = {i: self.Force[i] for i in self.Force.keys() if i < len(self.x) - k}

            Force_rates = {(i - len(self.x_process)): self.Force[i] for i in self.Force.keys() if
                           not i < len(self.x) - k}

        self.unpack_x_process(Force_process=Force_process, transformation=transformation)
        self.unpack_x_rates(Force_rates=Force_rates, transformation=transformation)

    def unpack_x_process(self, transformation, Force_process=None):
        if transformation == 'log':
            if self.noboundk==False:
               x_process = np.exp(self.x_process)
            else:
                x_process = np.exp(self.x_process)
                if self.ifmodel!="old":
                    if self.Model=="MG94":
                        x_process[6]=np.log(x_process[6])
                    else:
                        x_process[5] = np.log(x_process[5])

        elif transformation == 'None':
            x_process = self.x_process
        elif transformation == 'Exp_Neg':
            x_process = np.concatenate((self.x_process[:3], -np.log(self.x_process[3:])))

        if Force_process != None:
            for i in Force_process.keys():
                x_process[i] = Force_process[i]

        if self.Model == 'MG94':
            # x_process[] = %AG, %A, %C, kappa, tau, omega
            if self.ifmodel=="old":
                check_length = 6 + (not self.IGC_Omega is None)
            elif self.ifmodel=="EM_full":
                check_length = 7 + (not self.IGC_Omega is None)
            elif self.ifmodel=="EM_reduce":
                check_length = 2


            assert (len(self.x_process) == check_length)

            if self.ifmodel == "old" or self.ifmodel=="EM_full":
                pi_a = x_process[0] * x_process[1]
                pi_c = (1 - x_process[0]) * x_process[2]
                pi_g = x_process[0] * (1 - x_process[1])
                pi_t = (1 - x_process[0]) * (1 - x_process[2])
                self.pi = [pi_a, pi_c, pi_g, pi_t]
                self.kappa = x_process[3]
                self.omega = x_process[4]
                if self.IGC_Omega is None:
                    self.tau = x_process[5]
                    if self.ifmodel == "EM_full":
                        self.K = x_process[6]
                else:
                    self.IGC_Omega = x_process[5]
                    self.tau = x_process[6]
                    if self.ifmodel == "EM_full":
                        self.K = x_process[7]

            elif self.ifmodel == "EM_reduce":
                self.tau = x_process[0]
                self.K = x_process[1]


        elif self.Model == 'HKY':
            # x_process[] = %AG, %A, %C, kappa, tau
            if self.ifmodel=="old":
                check_length = 5
            elif self.ifmodel=="EM_full":
                check_length = 6
            elif self.ifmodel=="EM_reduce":
                check_length = 2

            assert (len(self.x_process) == check_length)

            if self.ifmodel == "old" or self.ifmodel=="EM_full":

                pi_a = x_process[0] * x_process[1]
                pi_c = (1 - x_process[0]) * x_process[2]
                pi_g = x_process[0] * (1 - x_process[1])
                pi_t = (1 - x_process[0]) * (1 - x_process[2])
                self.pi = [pi_a, pi_c, pi_g, pi_t]
                self.kappa = x_process[3]
                self.tau = x_process[4]
                if  self.ifmodel=="EM_full":
                    self.K = x_process[5]

            elif self.ifmodel == "EM_reduce":
                self.tau = x_process[0]
                self.K = x_process[1]


        if self.ifmodel == "old":

            # Now update the prior distribution
            self.get_prior()
            # Now update processes (Rate matrices)
            self.get_processes()
        else:
            self.get_prior()
            if self.Model=="MG94":
                 self.processes=self.Get_branch_Q(self.id)
            else:
                 self.processes = self.Get_branch_QHKY(self.id)



    def get_prior(self):
        if self.Model == 'MG94':
            self.prior_feasible_states = [(self.codon_to_state[codon], self.codon_to_state[codon]) for codon in
                                          self.codon_nonstop]
            distn = [reduce(mul, [self.pi['ACGT'.index(b)] for b in codon], 1) for codon in self.codon_nonstop]
            distn = np.array(distn) / sum(distn)
        elif self.Model == 'HKY':
            self.prior_feasible_states = [(self.nt_to_state[nt], self.nt_to_state[nt]) for nt in 'ACGT']
            distn = [self.pi['ACGT'.index(nt)] for nt in 'ACGT']
            distn = np.array(distn) / sum(distn)

        self.prior_distribution = distn

    def get_processes(self):
        if self.Model == 'MG94':
            self.processes = self.get_MG94Geneconv_and_MG94()
        elif self.Model == 'HKY':
            self.processes = self.get_HKYGeneconv()

    def get_save_file_name(self):
        if self.ifmodel=="old":
            if self.save_name is None:
                prefix_save = self.save_path + self.Model
                if not self.IGC_Omega is None:
                    prefix_save = prefix_save + '_twoOmega'
                if self.Force:
                    prefix_save = prefix_save + '_Force'

                ## if self.Dir:
                ##        prefix_save = prefix_save + '_Dir'
                ##
                ##if self.gBGC:
                ##        prefix_save = prefix_save + '_gBGC'

                if self.clock:
                    suffix_save = '_clock_save.txt'
                else:
                    suffix_save = '_nonclock_save.txt'

                save_file = prefix_save + '_' + '_'.join(self.paralog) + suffix_save
            else:
                save_file = self.save_name
        else:
            if self.save_name1 is None:
                prefix_save = self.save_path + self.Model
                if self.ifmodel != "old":
                    prefix_save = self.save_path + self.Model + self.ifmodel
                if not self.IGC_Omega is None:
                    prefix_save = prefix_save + '_twoOmega'
                if self.Force:
                    prefix_save = prefix_save + '_Force'

                if self.clock:
                    suffix_save = '_clock_save.txt'
                else:
                    suffix_save = '_nonclock_save.txt'

                save_file = prefix_save + '_' + '_'.join(self.paralog) + suffix_save
            else:
                save_file = self.save_name1

        return save_file

    def get_MG94Geneconv_and_MG94(self):
        Qbasic = self.get_MG94Basic()
        row = []
        col = []
        rate_geneconv = []
        rate_basic = []

        for i, pair in enumerate(product(self.codon_nonstop, repeat=2)):
            # use ca, cb, cc to denote codon_a, codon_b, codon_c, where cc != ca, cc != cb
            ca, cb = pair
            sa = self.codon_to_state[ca]
            sb = self.codon_to_state[cb]
            if ca != cb:
                for cc in self.codon_nonstop:
                    if cc == ca or cc == cb:
                        continue
                    sc = self.codon_to_state[cc]
                    # (ca, cb) to (ca, cc)
                    Qb = Qbasic[sb, sc]
                    if Qb != 0:
                        row.append((sa, sb))
                        col.append((sa, sc))
                        rate_geneconv.append(Qb)
                        rate_basic.append(0.0)

                    # (ca, cb) to (cc, cb)
                    Qb = Qbasic[sa, sc]
                    if Qb != 0:
                        row.append((sa, sb))
                        col.append((sc, sb))
                        rate_geneconv.append(Qb)
                        rate_basic.append(0.0)

                # (ca, cb) to (ca, ca)
                row.append((sa, sb))
                col.append((sa, sa))
                Qb = Qbasic[sb, sa]
                if isNonsynonymous(cb, ca, self.codon_table):
                    Tgeneconv = self.tau * self.get_IGC_omega()
                else:
                    Tgeneconv = self.tau
                rate_geneconv.append(Qb + Tgeneconv)
                rate_basic.append(0.0)

                # (ca, cb) to (cb, cb)
                row.append((sa, sb))
                col.append((sb, sb))
                Qb = Qbasic[sa, sb]
                rate_geneconv.append(Qb + Tgeneconv)
                rate_basic.append(0.0)

            else:
                for cc in self.codon_nonstop:
                    if cc == ca:
                        continue
                    sc = self.codon_to_state[cc]

                    # (ca, ca) to (ca,  cc)
                    Qb = Qbasic[sa, sc]
                    if Qb != 0:
                        row.append((sa, sb))
                        col.append((sa, sc))
                        rate_geneconv.append(Qb)
                        rate_basic.append(0.0)
                        # (ca, ca) to (cc, ca)
                        row.append((sa, sb))
                        col.append((sc, sa))
                        rate_geneconv.append(Qb)
                        rate_basic.append(0.0)

                        # (ca, ca) to (cc, cc)
                        row.append((sa, sb))
                        col.append((sc, sc))
                        rate_geneconv.append(0.0)
                        rate_basic.append(Qb)

        process_geneconv = dict(
            row=row,
            col=col,
            rate=rate_geneconv
        )
        process_basic = dict(
            row=row,
            col=col,
            rate=rate_basic
        )
        return [process_basic, process_geneconv]

    def get_MG94Basic(self):
        Qbasic = np.zeros((61, 61), dtype=float)
        for ca in self.codon_nonstop:
            for cb in self.codon_nonstop:
                if ca == cb:
                    continue
                Qbasic[self.codon_to_state[ca], self.codon_to_state[cb]] = get_MG94BasicRate(ca, cb, self.pi,
                                                                                             self.kappa, self.omega,
                                                                                             self.codon_table)

        expected_rate = np.dot(self.prior_distribution, Qbasic.sum(axis=1))
        Qbasic = Qbasic / expected_rate
        return Qbasic



    def get_HKYBasic(self):
        Qbasic = np.array([
            [0, 1.0, self.kappa, 1.0],
            [1.0, 0, 1.0, self.kappa],
            [self.kappa, 1.0, 0, 1.0],
            [1.0, self.kappa, 1.0, 0],
        ]) * np.array(self.pi)
        if np.dot(self.prior_distribution, Qbasic.sum(axis=1))!=0:
            expected_rate = np.dot(self.prior_distribution, Qbasic.sum(axis=1))
        else:
            expected_rate = 1
        Qbasic = Qbasic / expected_rate
        return Qbasic

    def get_HKYGeneconv(self):
        # print ('tau = ', self.tau)
        Qbasic = self.get_HKYBasic()
        row = []
        col = []
        rate_geneconv = []
        rate_basic = []

        for i, pair_from in enumerate(product('ACGT', repeat=2)):
            na, nb = pair_from
            sa = self.nt_to_state[na]
            sb = self.nt_to_state[nb]
            for j, pair_to in enumerate(product('ACGT', repeat=2)):
                nc, nd = pair_to
                sc = self.nt_to_state[nc]
                sd = self.nt_to_state[nd]
                if i == j:
                    continue
                GeneconvRate = get_HKYGeneconvRate(pair_from, pair_to, Qbasic, self.tau)
                if GeneconvRate != 0.0:
                    row.append((sa, sb))
                    col.append((sc, sd))
                    rate_geneconv.append(GeneconvRate)
                    rate_basic.append(0.0)
                if na == nb and nc == nd:
                    row.append((sa, sb))
                    col.append((sc, sd))
                    rate_geneconv.append(GeneconvRate)
                    rate_basic.append(Qbasic['ACGT'.index(na), 'ACGT'.index(nc)])

        process_geneconv = dict(
            row=row,
            col=col,
            rate=rate_geneconv
        )
        process_basic = dict(
            row=row,
            col=col,
            rate=rate_basic
        )
        # process_basic is for HKY_Basic which is equivalent to 4by4 rate matrix
        return [process_basic, process_geneconv]

    def unpack_x_rates(self, transformation,
                       Force_rates=None):  # TODO: Change it to fit general tree structure rather than cherry tree
        if transformation == 'log':
            x_rates = np.exp(self.x_rates)
        elif transformation == 'None':
            x_rates = self.x_rates
        elif transformation == 'Exp_Neg':
            x_rates = -np.log(self.x_rates)

        if Force_rates != None:
            for i in Force_rates.keys():
                x_rates[i] = Force_rates[i]
        assert (len(x_rates) == len(self.edge_to_blen))

        for edge_it in range(len(self.edge_list)):
            self.edge_to_blen[self.edge_list[edge_it]] = x_rates[edge_it]

        self.update_tree()

    def update_tree(self):
        for i in range(len(self.tree['rate'])):
            node1 = self.num_to_node[self.tree['row'][i]]
            node2 = self.num_to_node[self.tree['col'][i]]
            self.tree['rate'][i] = self.edge_to_blen[(node1, node2)]

    def _loglikelihood(self, store=True, edge_derivative=False):
        '''
        Modified from Alex's objective_and_gradient function in ctmcaas/adv-log-likelihoods/mle_geneconv_common.py
        '''
        if self.Model == 'MG94':
            state_space_shape = [61, 61]
        elif self.Model == 'HKY':
            state_space_shape = [4, 4]

        # prepare some extra parameters for the json interface
        if edge_derivative:
            requested_derivatives = list(range(k))
        else:
            requested_derivatives = []

        site_weights = np.ones(self.nsites)

        # prepare the input for the json interface
        data = dict(
            site_weights=site_weights,
            requested_derivatives=requested_derivatives,
            node_count=len(self.edge_to_blen) + 1,
            state_space_shape=state_space_shape,
            process_count=len(self.processes),
            processes=self.processes,
            tree=self.tree,
            prior_feasible_states=self.prior_feasible_states,
            prior_distribution=self.prior_distribution,
            observable_nodes=self.observable_nodes,
            observable_axes=self.observable_axes,
            iid_observations=self.iid_observations
        )
        j_ll = jsonctmctree.ll.process_json_in(data)

        status = j_ll['status']
        feasibility = j_ll['feasibility']

        if status != 'success' or not feasibility:
            print('results:')
            print(j_ll)
            print()
            raise Exception('Encountered some problem in the calculation of log likelihood and its derivatives')

        ll, edge_derivs = j_ll['log_likelihood'], j_ll['edge_derivatives']
        self.ll = ll

        return ll, edge_derivs

    def _loglikelihood2(self, store=True, edge_derivative=False):
        '''
        Modified from Alex's objective_and_gradient function in ctmcaas/adv-log-likelihoods/mle_geneconv_common.py
        '''



        if store:
            self.scene_ll = self.get_scene()
            scene = self.scene_ll
        else:
            scene = self.get_scene()

        log_likelihood_request = {'property': 'snnlogl'}
        derivatives_request = {'property': 'sdnderi'}
        if edge_derivative and self.ifmodel != "EM_reduce":
            requests = [log_likelihood_request, derivatives_request]
        else:
            requests = [log_likelihood_request]
        j_in = {
            'scene': self.scene_ll,
            'requests': requests
        }
        j_out = jsonctmctree.interface.process_json_in(j_in)

        status = j_out['status']


        ll = j_out['responses'][0]
        self.ll = ll
        if edge_derivative:
            edge_derivs = j_out['responses'][1]
            self.edge_de= edge_derivs
        else:
            edge_derivs = []

     #   print(edge_derivs)

        return ll, edge_derivs


    # def _loglikelihood3(self, store=True, edge_derivative=False):
    #         '''
    #         Modified from Alex's objective_and_gradient function in ctmcaas/adv-log-likelihoods/mle_geneconv_common.py
    #         '''
    #
    #
    #         self.update_by_x(self.x)
    #   #      print(self.x[4])
    #
    #         if store:
    #             self.scene_ll = self.get_scene()
    #             scene = self.scene_ll
    #         else:
    #             scene = self.get_scene()
    #
    #         log_likelihood_request = {'property': 'snnlogl'}
    #         derivatives_request = {'property': 'sdnderi'}
    #         if edge_derivative and self.ifmodel != "EM_reduce":
    #             requests = [log_likelihood_request, derivatives_request]
    #         else:
    #             requests = [log_likelihood_request]
    #         j_in = {
    #             'scene': self.scene_ll,
    #             'requests': requests
    #         }
    #         j_out = jsonctmctree.interface.process_json_in(j_in)
    #
    #         status = j_out['status']
    #
    #         try:
    #             ll = j_out['responses'][0]
    #         except Exception:
    #             print("xxxxxxxXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", flush=True)
    #             print(self.x, flush=True)
    #             print(edge_derivative, flush=True)
    #             print(self.edge_de, flush=True)
    #             print(self.scene_ll)
    #             print("xxxxxxxXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", flush=True)
    #             pass
    #
    #         #        ll = j_out['responses'][0]
    #         self.ll = ll
    #
    #         return ll

    def _sitewise_loglikelihood(self):
        scene = self.get_scene()

        log_likelihood_request = {'property': 'dnnlogl'}
        requests = [log_likelihood_request]

        j_in = {
            'scene': self.scene_ll,
            'requests': requests
        }
        j_out = jsonctmctree.interface.process_json_in(j_in)

        status = j_out['status']

        ll = j_out['responses'][0]
        self.ll = ll

        return ll

    def get_sitewise_loglikelihood_summary(self, summary_file):
        ll = self._sitewise_loglikelihood()
        with open(summary_file, 'w+') as f:
            f.write('#Site\tlnL\t\n')
            for i in range(self.nsites):
                f.write('\t'.join([str(i), str(ll[i])]) + '\n')

    def get_scene(self):

        if self.Model == 'MG94':
            state_space_shape = [61, 61]
        elif self.Model == 'HKY':
            state_space_shape = [4, 4]

        process_definitions = [{'row_states': i['row'], 'column_states': i['col'], 'transition_rates': i['rate']} for i
                               in self.processes]

        if self.ifmodel != "old":
            dd=[]
            for i in range(len(self.tree['col'])):
                    dd.append(i)
            self.tree['process']=dd

        scene = dict(
            node_count=len(self.edge_to_blen) + 1,
            process_count=len(self.processes),
            state_space_shape=state_space_shape,
            tree={
                'row_nodes': self.tree['row'],
                'column_nodes': self.tree['col'],
                'edge_rate_scaling_factors': self.tree['rate'],
                'edge_processes': self.tree['process']
            },
            root_prior={'states': self.prior_feasible_states,
                        'probabilities': self.prior_distribution},
            process_definitions=process_definitions,
            observed_data={
                'nodes': self.observable_nodes,
                'variables': self.observable_axes,
                'iid_observations': self.iid_observations
            }
        )

      #  print(scene)
        return scene

    def loglikelihood_and_gradient(self, package='new', display=False):
        '''
        Modified from Alex's objective_and_gradient function in ctmcaas/adv-log-likelihoods/mle_geneconv_common.py
        '''
        self.update_by_x()

        x = deepcopy(self.x)  # store the current x array
        if package == 'new':
            fn = self._loglikelihood2
        else:
            fn = self._loglikelihood

        if self.ifmodel=="old" or self.ifmodel=="EM_full":
               ll, edge_derivs = fn(edge_derivative=True)
        else:
                ll, edge_derivs = fn()

      #  print(ll)


        m = len(self.x) - len(self.edge_to_blen)

        # use finite differences to estimate derivatives with respect to these parameters
        other_derivs = []

        fix_x=deepcopy(np.array(self.x))



        for i in range(m):
            if self.Force != None:
                if i in self.Force.keys():  # check here
                    other_derivs.append(0.0)
                    continue


            delta=deepcopy(max(1,abs(self.x[i]))*0.000001)
            x_plus_delta1 = deepcopy(fix_x)
            x_plus_delta1[i] += delta/2
            self.update_by_x(x_plus_delta1)
            ll_delta1, _ = fn(store=True, edge_derivative=False)

 # ll_delta0 is f(x-h/2)
            x_plus_delta0 = deepcopy(fix_x)
            x_plus_delta0[i] -= delta/2
            self.update_by_x(x_plus_delta0)
            ll_delta0, _ = fn(store=True, edge_derivative=False)


            d_estimate = (ll_delta1 - ll_delta0) / delta

            other_derivs.append(d_estimate)
            # restore self.x
            self.update_by_x(x)

        other_derivs = np.array(other_derivs)



    #    print(ll)
        if display:
            print('log likelihood = ', ll, flush=True)
            print('Edge derivatives = ', edge_derivs, flush=True)
            print('other derivatives:', other_derivs, flush=True)
            print('Current x array = ', self.x, flush=True)

        self.ll = ll
        f = -ll
        g = -np.concatenate((other_derivs, edge_derivs))

      #  self.tau_F=self.tau
      #  self.k_F=self.K



        return f, g


    # def loglikelihood_and_gradient2(self, package='new', display=False):
    #     '''
    #     Modified from Alex's objective_and_gradient function in ctmcaas/adv-log-likelihoods/mle_geneconv_common.py
    #     '''
    #     self.update_by_x()
    #     delta = 1e-8
    #     x = deepcopy(self.x)  # store the current x array
    #     if package == 'new':
    #         fn = self._loglikelihood2
    #     else:
    #         fn = self._loglikelihood
    #
    #     ll, edge_derivs = fn(edge_derivative=True)
    #
    #     m = len(self.x) - len(self.edge_to_blen)
    #
    #     # use finite differences to estimate derivatives with respect to these parameters
    #     other_derivs = []
    #
    #     self.tau_F=self.tau
    #     self.k_F=self.K
    #
    #     for i in range(m):
    #         if self.Force != None:
    #             if i in self.Force.keys():  # check here
    #                 other_derivs.append(0.0)
    #                 continue
    #         x_plus_delta = np.array(self.x)
    #         x_plus_delta[i] += delta / 2.0
    #         self.update_by_x(x_plus_delta)
    #         ll_delta_plus, _ = fn(store=True, edge_derivative=False)
    #         x_plus_delta[i] -= delta
    #         self.update_by_x(x_plus_delta)
    #         ll_delta_minus, _ = fn(store=True, edge_derivative=False)
    #
    #
    #         d_estimate = (ll_delta_plus - ll_delta_minus) / delta
    #         other_derivs.append(d_estimate)
    #         # restore self.x
    #         self.update_by_x(x)
    #     other_derivs = np.array(other_derivs)
    #
    #     if display:
    #         print('log likelihood = ', ll,flush=True)
    #         print('Edge derivatives = ', edge_derivs,flush=True)
    #         print('other derivatives:', other_derivs,flush=True)
    #         print('Current x array = ', self.x,flush=True)
    #
    #
    #     self.ll = ll
    #     f = -ll
    #     g = -np.concatenate((other_derivs, edge_derivs))
    #
    #     self.tau_F=self.tau
    #     self.k_F=self.K
    #
    #     return f, g

    def objective_and_gradient(self, display, x):
        self.update_by_x(x)


        f, g = self.loglikelihood_and_gradient(display=display)
        self.auto_save += 1
        if self.auto_save == 3:
           self.save_x()
           self.auto_save = 0

        return f, g

    def objective_and_gradient_EM_full(self, display, x):

        self.update_by_x(x)
        f, g = self.loglikelihood_and_gradient(display=display)
        self.auto_save1 += 1
        if self.auto_save1 == 1:
            self.save_x()
            self.auto_save1 = 0
        return f, g

    def objective_and_gradient_EM_reduce(self, display, x):
        self.update_by_x(x)
        f, g = self.loglikelihood_and_gradient(display=display)
        self.auto_save += 1
        return f, g

    def Clock_wrap(self, display, x_clock):
        assert (self.clock)
        self.update_by_x_clock(x_clock)

        f, g = self.loglikelihood_and_gradient()

        # Now need to calculate the derivatives
        nEdge = len(self.edge_to_blen)  # number of edges
        l = nEdge / 2 + 1  # number of leaves
        k = l - 1  # number of internal nodes. The notation here is inconsistent with Alex's for trying to match my notes.

        other_derives, edge_derives = g[:-nEdge], g[-nEdge:]
        edge_to_derives = {self.edge_list[i]: edge_derives[i] for i in range(len(self.edge_list))}

        leaf_branch = [edge for edge in self.edge_to_blen.keys() if
                       edge[0][0] == 'N' and str.isdigit(edge[0][1:]) and not str.isdigit(edge[1][1:])]
        out_group_branch = [edge for edge in leaf_branch if edge[0] == 'N0' and not str.isdigit(edge[1][1:])][0]
        internal_branch = [x for x in self.edge_to_blen.keys() if not x in leaf_branch]
        assert (len(
            internal_branch) == k - 1)  # check if number of internal branch is one less than number of internal nodes

        leaf_branch.sort(
            key=lambda node: int(node[0][1:]))  # sort the list by the first node number in increasing order
        internal_branch.sort(
            key=lambda node: int(node[0][1:]))  # sort the list by the first node number in increasing order

        Lr_derives = []  # used to store derivatives for the clock parameters L, r0, r1, ...
        Lr_derives.append(sum(edge_derives))  # dLL/dL = sum(all derives)
        Lr_derives.append(edge_to_derives[out_group_branch] * 2 / (self.x_Lr[1] - 2)
                          + sum(edge_derives))

        for i in range(2, len(self.x_Lr)):  # r(i-1)
            if self.x_Lr[i] < 1:  # when no inf could happen in the transformation
                Lr_derives.append(
                    edge_to_derives[('N' + str(i - 2), 'N' + str(i - 1))] * (self.x_Lr[i] / (self.x_Lr[i] - 1))  #
                    + sum([edge_to_derives[internal_branch[j]] for j in
                           range(i - 1, len(internal_branch))])  # only sum over nodes decendent from node i-1
                    + sum([edge_to_derives[leaf_branch[j]] for j in
                           range(i - 1, len(leaf_branch))]))  # only sum over nodes decendent from node i-1
            else:  # get numerical derivative instead when inf happens
                ll = self._loglikelihood2()[0]
                self.x_clock[i + len(other_derives)] += 1e-8
                self.update_by_x_clock()
                l = self._loglikelihood2()[0]
                Lr_derives.append((l - ll) / 1e-8)
                self.x_clock[i + len(other_derives)] -= 1e-8
                self.update_by_x_clock()

        # TODO: Need to change the two sums if using general tree

        g_clock = np.concatenate((np.array(other_derives), np.array(Lr_derives)))

        if display:
            print('log likelihood = ', -f)
            print('Lr derivatives = ', Lr_derives)
            print('other derivatives = ', other_derives)
            print('Current x_clock array = ', self.x_clock)

        return f, g_clock

    def objective_wo_derivative(self, x):
        if self.clock:
            self.update_by_x_clock(x)
            ll = self._loglikelihood2()[0]
        else:
            self.update_by_x(x)
            ll = self._loglikelihood2()[0]


        # if display:
        #    print('log likelihood = ', ll)
        #    if self.clock:
        #        print('Current x_clock array = ', self.x_clock)
        #    else:
        #        print('Current x array = ', self.x)

        return -ll

# develop this one to compute hessian
    def objective_wo_derivative1(self, x):

        if self.ifexp==False:
            if self.Model == "MG94":
                self.x[5] = x[0]
                self.x[6] = x[1]

            else:
                self.x[4] = x[0]
                self.x[5] = x[1]
            self.update_by_x(self.x)
            self.id = self.compute_paralog_id()
            self.update_by_x(self.x)

            ll = self._loglikelihood2()[0]

        else:
            if self.Model == "MG94":

                if x[0]>0:
                    self.x[5] = np.log(x[0])
                else:
                    self.x[5] = np.log(0.0001)

                self.x[6] = x[1]
            else:
                if x[0]>0:
                    self.x[4] = np.log(x[0])
                else:
                    self.x[4] = np.log(0.0001)
                self.x[5] = x[1]

            self.update_by_x(self.x)
            self.id = self.compute_paralog_id()
            self.update_by_x(self.x)
            ll = self._loglikelihood2()[0]


        return -ll

    def objective_wo_derivative_global(self, display, x):
        if self.clock:
            self.update_by_x_clock(x, transformation='Exp_Neg')
            ll = self._loglikelihood2()[0]
        else:
            self.update_by_x(x, transformation='Exp_Neg')
            ll = self._loglikelihood2()[0]

        if display:
            print('log likelihood = ', ll)
            if self.clock:
                print('Current x_clock array = ', self.x_clock)
            else:
                print('Current x array = ', self.x)

        return -ll

    def get_mle(self, display=True, derivative=True, em_iterations=0, method='BFGS', niter=2000,
                tauini=1.1,omegaini=1.4,kini=1.1,ifseq=False):

        if ifseq==True and self.Model=="HKY":

            self.tau=tauini
            self.x_process[4]=np.log(tauini)
            self.x[4] = np.log(tauini)
            if self.ifmodel!="old":
                self.x[5]=kini

        #    print(self.x)
       #     print(self.tau)

        if ifseq==True and self.Model=="MG94":

            self.tau=tauini
            self.omega=omegaini
            self.x_process[5]=np.log(tauini)
            self.x[5] = np.log(tauini)
            self.x_process[4]=np.log(omegaini)
            self.x[4] = np.log(omegaini)
            if self.ifmodel != "old":
                self.x[6] = kini

        if em_iterations > 0:
            ll = self._loglikelihood2()
            # http://jsonctmctree.readthedocs.org/en/latest/examples/hky_paralog/yeast_geneconv_zero_tau/index.html#em-for-edge-lengths-only
            observation_reduction = None
            self.x_rates = np.log(optimize_em(self.get_scene, observation_reduction, em_iterations))
            self.x = np.concatenate((self.x_process, self.x_rates))
            if self.clock:
                self.update_x_clock_by_x()
                self.update_by_x_clock()
            else:
                self.update_by_x()

            if display:
                print('log-likelihood = ', ll)
                print('updating blen length using EM')
                print('current log-likelihood = ', self._loglikelihood2())
        else:
            if self.clock:
                self.update_by_x_clock()
            else:
                self.update_by_x()

        bnds = [(-8.0, -0.05)] * 3
        if not self.clock:
            self.update_by_x()
            if derivative:
                if self.ifmodel=="old":
                   f = partial(self.objective_and_gradient, display)

                elif self.ifmodel=="EM_full":
                   f = partial(self.objective_and_gradient_EM_full, display)
                elif self.ifmodel=="EM_reduce":
                   f = partial(self.objective_and_gradient_EM_reduce, display)

            else:
                f = partial(self.objective_wo_derivative, display)
            guess_x = self.x
            if self.ifmodel=="old" :
                bnds.extend([(None, 6.0)] * (len(self.x_process) - 4))
                edge_bnds = [(None, 4.0)] * len(self.x_rates)
                edge_bnds[1] = (self.minlogblen, 4.0)
            elif self.ifmodel=="EM_full":
                bnds.extend([(-10.0, 6.0)] * (len(self.x_process) - 5))
                edge_bnds = [(-10.0, 4.0)] * len(self.x_rates)
                edge_bnds[1] = (self.minlogblen, None)
            else:
                bnds = [(None, 7.0)] * 1
                bnds.extend([(None, 20)] * 1)

#tau and K
            if self.ifmodel=="old":
                bnds.extend([(-10.0, 6.0)] * (1))

            if self.ifmodel=="EM_full":
                if self.bound == True:
                    low = np.log(deepcopy(self.tau)) - 1
                    high = np.log(deepcopy(self.compute_bound()))
                    bnds.extend([(low, high)] * (1))
                    khigh = np.log(deepcopy(float(self.kbound)))
                    bnds.extend([(-4, khigh)] * (1))
                else:
                    bnds.extend([(-10.0, 7.0)] * (1))
                    if self.noboundk==True:
                        bnds.extend([(-10.0, 80.0)] * (1))
                    else:
                        bnds.extend([(-10.0, 7)] * (1))


            bnds.extend(edge_bnds)

        else:
            self.update_by_x_clock()  # TODO: change force for blen in x_clock
            if derivative:
                f = partial(self.Clock_wrap, display)
            else:
                f = partial(self.objective_wo_derivative, display)
            guess_x = self.x_clock
            assert (len(self.edge_to_blen) % 2 == 0)
            l = int(len(self.edge_to_blen) / 2)
            bnds.extend([(None, None)] * (len(self.x_clock) - 2 - (l + 1)))
            bnds.extend([(-10, 0.0)] * l)
        if method == 'BFGS':
            if derivative:

            #    if self.ifmodel=="old":

                       result = scipy.optimize.minimize(f, guess_x, jac=True, method='L-BFGS-B', bounds=bnds)
        #        else:

           #            result = scipy.optimize.basinhopping(f, guess_x,
            #                                       minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True, 'bounds': bnds},
            #                                       niter=100)
            else:
                result = scipy.optimize.minimize(f, guess_x, jac=False, method='L-BFGS-B', bounds=bnds)
        elif method == 'differential_evolution':
            f = partial(self.objective_wo_derivative_global, display)
            if self.clock:
                bnds = [(np.exp(-20), 1.0 - np.exp(-20))] * len(self.x_clock)
            else:
                bnds = [(np.exp(-20), 1.0 - np.exp(-20))] * len(self.x)

            result = scipy.optimize.differential_evolution(f, bnds, callback=self.check_boundary_differential_evolution)
        elif method == 'basin-hopping':
            if derivative:
                result = scipy.optimize.basinhopping(f, guess_x, minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True,
                                                                                   'bounds': bnds},
                                                     niter=niter)  # , callback = self.check_boundary)
            else:
                result = scipy.optimize.basinhopping(f, guess_x, minimizer_kwargs={'method': 'L-BFGS-B', 'jac': False,
                                                                                   'bounds': bnds},
                                                     niter=niter)  # , callback = self.check_boundary)

        if ifseq != True:
            print(result)



        self.save_x()

        return result

    def check_boundary(self, x, f, accepted):
        print("at minimum %.4f accepted %d" % (f, int(accepted)))
        return self.edge_to_blen[self.edge_list[1]] > np.exp(self.minlogblen)

    def save_x(self):
        if self.clock:
            save = self.x_clock
        else:
            save = self.x

        save_file = self.get_save_file_name()
      #  print(save_file)

        np.savetxt(save_file, save.T)

    def check_boundary_differential_evolution(self, x, convergence):
        print("at lnL %.4f convergence fraction %d" % (self.ll, convergence))
        # return self.edge_to_blen[self.edge_list[1]] > np.exp(self.minlogblen)

    def update_x_clock_by_x(self):
        Lr = []
        x_rates = np.exp(self.x_rates)
        for i in range(len(self.edge_list) / 2 - 1):
            elist = {self.edge_list[a]: x_rates[a] for a in range(len(self.edge_list)) if
                     self.edge_list[a][0] == 'N' + str(i)}
            elenlist = [elist[t] for t in elist]
            if i == 0:
                extra_list = [x_rates[a] for a in range(len(self.edge_list)) if
                              self.edge_list[a][0] == 'N' + str(1) and self.edge_list[a][1][0] != 'N']
                L = (sum(elenlist) + extra_list[0]) / 2
                r0 = 2.0 - (sum(elenlist) - elist[('N0', 'N1')]) / L
                Lr.append(L)
                Lr.append(r0)
                Lr.append(extra_list[0] / (L * r0))
            else:
                Lr.append(1 - min(elenlist) / max(elenlist))
        self.x_Lr = np.array(Lr)
        self.x_clock = np.concatenate((self.x_process, np.log(self.x_Lr)))

    def get_IGC_omega(self):
        if self.IGC_Omega is None:
            omega = self.omega
        else:
            omega = self.IGC_Omega
        return omega



    def isSynonymous(self, first_codon, second_codon):
        return self.codon_table[first_codon] == self.codon_table[second_codon]

    def isNonsynonymous(self, ca, cb, codon_table):
        return (codon_table[ca] != codon_table[cb])

    def compute_bound(self):
        self.id[1]=1
        min = self.id[0];

        # Loop through the array
        for i in range(len(self.id)):
            if (self.id[i] < min):
                min = self.id[i]

        taubound=self.tau/np.power(min,self.kbound)
        self.id[1] = 0

        return taubound



# here is for MG94
    def Get_branch_Q(self,paralog_id):

        Qbasic = self.get_MG94Basic()

        rate_basic = []
        Qlist = []
        ttt = len(self.tree['col'])


        for branch in range(ttt):
            row = []
            col = []

            if branch == 0:
                  rate_geneconv = []
                  for i, pair in enumerate(product(self.codon_nonstop, repeat=2)):
                # use ca, cb, cc to denote codon_a, codon_b, codon_c, where cc != ca, cc != cb
                     ca, cb = pair
                     sa = self.codon_to_state[ca]
                     sb = self.codon_to_state[cb]
                     if ca != cb:
                        for cc in self.codon_nonstop:
                            if cc == ca or cc == cb:
                                continue
                            sc = self.codon_to_state[cc]
                            # (ca, cb) to (ca, cc)
                            Qb = Qbasic[sb, sc]
                            if Qb != 0:
                                row.append((sa, sb))
                                col.append((sa, sc))
                                rate_geneconv.append(Qb)

                            # (ca, cb) to (cc, cb)
                            Qb = Qbasic[sa, sc]
                            if Qb != 0:
                                row.append((sa, sb))
                                col.append((sc, sb))
                                rate_geneconv.append(Qb)

                        # (ca, cb) to (ca, ca)
                        row.append((sa, sb))
                        col.append((sa, sa))
                        Qb = Qbasic[sb, sa]



                        if isNonsynonymous(cb, ca, self.codon_table):
                            Tgeneconv = self.tau * np.power(paralog_id[branch], self.K) *self.get_IGC_omega()
                        else:
                            Tgeneconv = self.tau * np.power(paralog_id[branch], self.K)
                        rate_geneconv.append(Qb + Tgeneconv)

                        # (ca, cb) to (cb, cb)
                        row.append((sa, sb))
                        col.append((sb, sb))
                        Qb = Qbasic[sa, sb]
                        rate_geneconv.append(Qb + Tgeneconv)

                     else:
                        for cc in self.codon_nonstop:
                            if cc == ca:
                                continue
                            sc = self.codon_to_state[cc]

                            # (ca, ca) to (ca,  cc)
                            Qb = Qbasic[sa, sc]
                            if Qb != 0:
                                row.append((sa, sb))
                                col.append((sa, sc))
                                rate_geneconv.append(Qb)
                                # (ca, ca) to (cc, ca)
                                row.append((sa, sb))
                                col.append((sc, sa))
                                rate_geneconv.append(Qb)

                                # (ca, ca) to (cc, cc)
                                row.append((sa, sb))
                                col.append((sc, sc))
                                rate_geneconv.append(0.0)

                  process_geneconv = dict(
                       row=deepcopy(row),
                       col=deepcopy(col),
                       rate=deepcopy(rate_geneconv)
                  )

                  Qlist.append(deepcopy(process_geneconv))

            elif branch > 1:
                  rate_geneconv = []
                  for i, pair in enumerate(product(self.codon_nonstop, repeat=2)):
                # use ca, cb, cc to denote codon_a, codon_b, codon_c, where cc != ca, cc != cb
                     ca, cb = pair
                     sa = self.codon_to_state[ca]
                     sb = self.codon_to_state[cb]
                     if ca != cb:
                        for cc in self.codon_nonstop:
                            if cc == ca or cc == cb:
                                continue
                            sc = self.codon_to_state[cc]
                            # (ca, cb) to (ca, cc)
                            Qb = Qbasic[sb, sc]
                            if Qb != 0:
                                row.append((sa, sb))
                                col.append((sa, sc))
                                rate_geneconv.append(Qb)

                            # (ca, cb) to (cc, cb)
                            Qb = Qbasic[sa, sc]
                            if Qb != 0:
                                row.append((sa, sb))
                                col.append((sc, sb))
                                rate_geneconv.append(Qb)

                        # (ca, cb) to (ca, ca)
                        row.append((sa, sb))
                        col.append((sa, sa))
                        Qb = Qbasic[sb, sa]
                        if isNonsynonymous(cb, ca, self.codon_table):
                            Tgeneconv = self.tau *np.power(paralog_id[branch],self.K)*self.get_IGC_omega()
                        else:
                            Tgeneconv = self.tau*np.power(paralog_id[branch],self.K)
                        rate_geneconv.append(Qb + Tgeneconv)

                        # (ca, cb) to (cb, cb)
                        row.append((sa, sb))
                        col.append((sb, sb))
                        Qb = Qbasic[sa, sb]
                        rate_geneconv.append(Qb + Tgeneconv)

                     else:
                        for cc in self.codon_nonstop:
                            if cc == ca:
                                continue
                            sc = self.codon_to_state[cc]

                            # (ca, ca) to (ca,  cc)
                            Qb = Qbasic[sa, sc]
                            if Qb != 0:
                                row.append((sa, sb))
                                col.append((sa, sc))
                                rate_geneconv.append(Qb)
                                # (ca, ca) to (cc, ca)
                                row.append((sa, sb))
                                col.append((sc, sa))
                                rate_geneconv.append(Qb)

                                # (ca, ca) to (cc, cc)
                                row.append((sa, sb))
                                col.append((sc, sc))
                                rate_geneconv.append(0.0)

                  process_geneconv = dict(
                       row=deepcopy(row),
                       col=deepcopy(col),
                       rate=deepcopy(rate_geneconv)
                  )

                  Qlist.append(deepcopy(process_geneconv))

            else:
                for i, pair in enumerate(product(self.codon_nonstop, repeat=2)):
                    # use ca, cb, cc to denote codon_a, codon_b, codon_c, where cc != ca, cc != cb
                    ca, cb = pair
                    sa = self.codon_to_state[ca]
                    sb = self.codon_to_state[cb]
                    if ca != cb:
                        for cc in self.codon_nonstop:
                            if cc == ca or cc == cb:
                                continue
                            sc = self.codon_to_state[cc]
                            # (ca, cb) to (ca, cc)
                            Qb = Qbasic[sb, sc]
                            if Qb != 0:
                                row.append((sa, sb))
                                col.append((sa, sc))
                                rate_basic.append(0.0)

                            # (ca, cb) to (cc, cb)
                            Qb = Qbasic[sa, sc]
                            if Qb != 0:
                                row.append((sa, sb))
                                col.append((sc, sb))
                                rate_basic.append(0.0)

                        # (ca, cb) to (ca, ca)
                        row.append((sa, sb))
                        col.append((sa, sa))
                        rate_basic.append(0.0)

                        # (ca, cb) to (cb, cb)
                        row.append((sa, sb))
                        col.append((sb, sb))
                        rate_basic.append(0.0)

                    else:
                        for cc in self.codon_nonstop:
                            if cc == ca:
                                continue
                            sc = self.codon_to_state[cc]

                            # (ca, ca) to (ca,  cc)
                            Qb = Qbasic[sa, sc]
                            if Qb != 0:
                                row.append((sa, sb))
                                col.append((sa, sc))
                                rate_basic.append(0.0)
                                # (ca, ca) to (cc, ca)
                                row.append((sa, sb))
                                col.append((sc, sa))
                                rate_basic.append(0.0)

                                # (ca, ca) to (cc, cc)
                                row.append((sa, sb))
                                col.append((sc, sc))
                                rate_basic.append(Qb)

                process_basic = dict(
                    row=deepcopy(row),
                    col=deepcopy(col),
                    rate=rate_basic
                )
                Qlist.append(deepcopy(process_basic))


        return Qlist

    def Get_branch_QHKY(self,paralog_id):

        Qbasic = self.get_HKYBasic()

        rate_basic = []
        Qlist = []
        ttt = len(self.tree['col'])


        for branch in range(ttt):
            row = []
            col = []

            if branch == 0:
                  rate_geneconv = []
                  for i, pair_from in enumerate(product('ACGT', repeat=2)):
                      na, nb = pair_from
                      sa = self.nt_to_state[na]
                      sb = self.nt_to_state[nb]
                      for j, pair_to in enumerate(product('ACGT', repeat=2)):
                          nc, nd = pair_to
                          sc = self.nt_to_state[nc]
                          sd = self.nt_to_state[nd]
                          if i == j:
                              continue
                          GeneconvRate = get_HKYGeneconvRate(pair_from, pair_to, Qbasic, self.tau*np.power(paralog_id[branch], self.K))
                          if GeneconvRate != 0.0:
                              row.append((sa, sb))
                              col.append((sc, sd))
                              rate_geneconv.append(GeneconvRate)

                          if na == nb and nc == nd:
                              row.append((sa, sb))
                              col.append((sc, sd))
                              rate_geneconv.append(GeneconvRate)


                  process_geneconv = dict(
                       row=deepcopy(row),
                       col=deepcopy(col),
                       rate=deepcopy(rate_geneconv)
                  )

                  Qlist.append(deepcopy(process_geneconv))

            elif branch > 1:
                rate_geneconv = []
                for i, pair_from in enumerate(product('ACGT', repeat=2)):
                    na, nb = pair_from
                    sa = self.nt_to_state[na]
                    sb = self.nt_to_state[nb]
                    for j, pair_to in enumerate(product('ACGT', repeat=2)):
                        nc, nd = pair_to
                        sc = self.nt_to_state[nc]
                        sd = self.nt_to_state[nd]
                        if i == j:
                            continue
                        GeneconvRate = get_HKYGeneconvRate(pair_from, pair_to, Qbasic,
                                                           self.tau * np.power(paralog_id[branch], self.K)
                                                           )
                        if GeneconvRate != 0.0:
                            row.append((sa, sb))
                            col.append((sc, sd))
                            rate_geneconv.append(GeneconvRate)
                        if na == nb and nc == nd:
                            row.append((sa, sb))
                            col.append((sc, sd))
                            rate_geneconv.append(GeneconvRate)

                process_geneconv = dict(
                    row=deepcopy(row),
                    col=deepcopy(col),
                    rate=deepcopy(rate_geneconv)
                )

                Qlist.append(deepcopy(process_geneconv))


            else:
                rate_geneconv = []

                for i, pair_from in enumerate(product('ACGT', repeat=2)):
                    na, nb = pair_from
                    sa = self.nt_to_state[na]
                    sb = self.nt_to_state[nb]
                    for j, pair_to in enumerate(product('ACGT', repeat=2)):
                        nc, nd = pair_to
                        sc = self.nt_to_state[nc]
                        sd = self.nt_to_state[nd]
                        if i == j:
                            continue
                        GeneconvRate = get_HKYGeneconvRate(pair_from, pair_to, Qbasic, self.tau)
                        if GeneconvRate != 0.0:
                            row.append((sa, sb))
                            col.append((sc, sd))
                            rate_geneconv.append(GeneconvRate)
                            rate_basic.append(0.0)
                        if na == nb and nc == nd:
                            row.append((sa, sb))
                            col.append((sc, sd))
                            rate_geneconv.append(GeneconvRate)
                            rate_basic.append(Qbasic['ACGT'.index(na), 'ACGT'.index(nc)])


                process_basic = dict(
                    row=deepcopy(row),
                    col=deepcopy(col),
                    rate=rate_basic
                )
                Qlist.append(deepcopy(process_basic))


        return Qlist


    def jointly_common_ancstral_inference(self):



        self.ancestral_state_response = deepcopy(self.get_ancestral_state_response_x())

        self.node_length = len(self.num_to_node)
        sites = np.zeros(shape=(self.node_length, self.nsites))
        for site in range(self.nsites):
            for node in range(self.node_length):
                sites[node][site] = int(np.array(self.ancestral_state_response[site])[node])

        self.sites = sites

    def get_ancestral_state_response_x(self):

        scene=self.get_scene()

        requests = [
            {'property': "ddnance"}
        ]
        # ref:https://jsonctmctree.readthedocs.io/en/latest/examples/yang2014/section_4_4_2_1_marginal/main.html
        if isinstance(scene, list):  # ref: _loglikelihood() function in JSGeneconv.py
            raise RuntimeError('Not yet tested.')
            separate_j_out = []
            for separate_scene in scene:
                j_in = {
                    'scene': separate_scene,
                    "requests": requests
                }
                j_out = jsonctmctree.interface.process_json_in(j_in)
                separate_j_out.append(j_out)
            result = separate_j_out

        else:
            j_in = {
                'scene': scene,
                'requests': requests
            }
            j_out = jsonctmctree.interface.process_json_in(j_in, debug = True)
            if j_out['status'] == 'feasible':
                result = j_out['responses'][0]
            else:
                raise RuntimeError('Failed at obtaining ancestral state distributions.')

        return result

    def difference(self,ini,ifdnalevel=False):
        index = 0
        ratio_nonsynonymous = 0
        ratio_synonymous = 0


        if self.Model=="HKY":
            str = {0, 5, 10, 15}

            for i in range(self.nsites):
                if not ini[i] in str:
                    index=index+1
        else:
            if ifdnalevel==False:
                for i in range(self.nsites):
                    ca=(ini[i])//61
                    cb=(ini[i])%61
                    if ca != cb:
                        index=index+1
                        cb1=self.state_to_codon[cb]
                        ca1 = self.state_to_codon[ca]
                        if self.isNonsynonymous(cb1, ca1, self.codon_table):
                            ratio_nonsynonymous=ratio_nonsynonymous+1

                if index != 0:
                        ratio_nonsynonymous = ratio_nonsynonymous / index
                        ratio_synonymous = 1 - ratio_synonymous

            else:
                for i in range(self.nsites):
                    ca = (ini[i]) // 61
                    cb = (ini[i]) % 61

                    cb1 = self.state_to_codon[cb]
                    ca1 = self.state_to_codon[ca]
                    for ii in range(3):
                        if cb1[ii]!=ca1[ii]:
                            index=index+1

                index=index/3

        return index,ratio_nonsynonymous,ratio_synonymous


    def compute_paralog_id(self,repeat=5,ifdnalevel=False):

        ttt = len(self.tree['col'])
        id = np.zeros(ttt)
        diverge_list = np.ones(ttt)
        if self.sites is None:
            self.jointly_common_ancstral_inference()

        if self.dwell_id == False:

            for mc in range(repeat):
                print("ddddd")
                self.jointly_common_ancstral_inference()
                for j in range(ttt):
                    if self.Model == "HKY":
                        if not j == 1:
                            ini2 = self.node_to_num[self.edge_list[j][0]]
                            end2 = self.node_to_num[self.edge_list[j][1]]
                            ini1 = deepcopy(self.sites[ini2,])
                            end1 = deepcopy(self.sites[end2,])
                            diverge_list[j] = diverge_list[j] + (self.difference(ini1)[0] + self.difference(end1)[0]) * 0.5
                            print("ccccccccccccc")
                            print("ddddd")
                            print(1-self.difference(ini1)[0] / self.nsites)
                            print(1-self.difference(end1)[0]/self.nsites)
                            print("ddddd")
                            print("ccccccccccccc")

                    if self.Model == "MG94":
                        if not j == 1:
                            ini2 = self.node_to_num[self.edge_list[j][0]]
                            end2 = self.node_to_num[self.edge_list[j][1]]
                            ini1 = deepcopy(self.sites[ini2,])
                            end1 = deepcopy(self.sites[end2,])
                            ini_D = self.difference(ini1,ifdnalevel=ifdnalevel)[0]
                            end_D = self.difference(end1,ifdnalevel=ifdnalevel)[0]
                            diverge_list[j] = diverge_list[j] + (float(ini_D + end_D) * 0.5)

            for j in range(ttt):
                if not j == 1:
                    id[j] = 1-(float(diverge_list[j]) / repeat)/self.nsites

        else:
            if self.Model == "MG94":
                if self.ifDNA==True:
                    expected_DwellTime = self._ExpectedHetDwellTime_DNA()

                    id = [ 1-(((
                         expected_DwellTime[0][i] +  expected_DwellTime[1][i])+
                           (  expected_DwellTime[2][i] +  expected_DwellTime[3][i])*2 +
                          ( expected_DwellTime[4][i] +  expected_DwellTime[5][i]) * 3)
                          /(2*3*self.nsites))
                           for i in range(ttt)]
                else:
                    expected_DwellTime = self._ExpectedHetDwellTime()
                    id = [1 - ((
                                       expected_DwellTime[0][i] +  expected_DwellTime[1][i])
                               / (2 * self.nsites))
                          for i in range(ttt)]

            else:
                expected_DwellTime = self._ExpectedHetDwellTime()
                id = [1 - (expected_DwellTime[i] / ( 2*self.nsites
                                                    ))
                      for i in range(ttt)]

        self.id=id

        return id

    def initialize_by_save(self, save_file):

        if self.clock:
            self.x_clock = np.loadtxt(open(save_file, 'r'))
            self.update_by_x_clock()
        else:
            self.x = np.loadtxt(open(save_file, 'r'))
            self.update_by_x()


    def EM_branch_tau(self,MAX=8,epis=0.01,force=None,K=0.5,bound=False,ifdnalevel=False):
        list=[]
        list.append(self.nsites)
        ll0=self.get_mle()["fun"]
        pstau=deepcopy(self.tau)
        self.id = self.compute_paralog_id(ifdnalevel=ifdnalevel)
        print(self.id)
        idold=deepcopy(self.id)
        print(self.get_Hessian())
        old_sum = self.get_summary(branchtau=True)
        self.K=K
        self.Force=force
        self.ifmodel = "EM_full"
        self.get_initial_x_process()
        self.ini_by_file()
        self.bound=bound
        self.get_mle()
        difference=abs(self.tau-pstau)

        print("EMcycle:")
        print(0)
        print(self.id)
        print(self.K)
        print(self.tau)
        print("xxxxxxxxxxxxxxxxx")
        print("xxxxxxxxxxxxxxxxx")
        print("xxxxxxxxxxxxxxxxx")
        print("\n")

        i=1
        while i<=MAX and difference >=epis:
            pstau_1 = deepcopy(self.tau)
            self.id = self.compute_paralog_id(ifdnalevel=ifdnalevel)
            ll1=self.get_mle()["fun"]
            difference = abs(self.tau - pstau_1)

            print("EMcycle:")
            print(i)
            i = i + 1
            print(self.id)
            print("K:")
            print(self.K)
            print("Tau:")
            print(self.tau)
            print("xxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxx")
            print("\n")


        new_sum = self.get_summary(branchtau=True)


        print("old tau: ",pstau)
        list.append(pstau)
        print("old ll: ",ll0)
        print("old sum IGC",old_sum[0])
        print("old sum TOTAL", old_sum[1])
        list.append(old_sum[0])
        list.append(old_sum[1])
        print("old igc prop",old_sum[0]/old_sum[1])
        list.append(old_sum[0]/old_sum[1])
        for j in range(len(self.edge_list)):
                list.append(idold[j])

        for j in range(len(self.edge_list)):
                list.append(old_sum[2][j])

        print("new sum IGC",new_sum[0])
        print("new sum TOTAL", new_sum[1])
        print("new tau: ",self.tau)
        print("K: ", self.K)
        print("new ll: ",ll1)
        list.append(ll1)
        list.append(self.tau)
        list.append(self.K)
        list.append(new_sum[0])
        list.append(new_sum[1])
        list.append(new_sum[0] / new_sum[1])
        print("new igc prop", new_sum[0] / new_sum[1])


        for j in range(len(self.edge_list)):
                list.append(self.id[j])

        for j in range(len(self.edge_list)):
                list.append(new_sum[2][j])

        hessian=self.get_Hessian()
        print(hessian)
        list.append(hessian[0][0])
        list.append(hessian[1][1])
        self.ifexp=True
        hessian = self.get_Hessian()
        print(hessian)
        list.append(hessian[0][0])
        list.append(hessian[1][1])



        save_nameP = "./save/" + self.Model +'.txt'
        with open(save_nameP, 'wb') as f:
            np.savetxt(f, list)

        print(list)







# this function is used to renew ini
    def renew_em_joint(self,ifmodel="EM_full",ifdnalevel=False):
        self.id = self.compute_paralog_id(ifdnalevel=ifdnalevel)
        print(self.id)
        self.ifmodel=ifmodel
        self.noboundk=True
        self.get_initial_x_process()


    def get_Hessian(self,basicstep=None):

        if self.ifexp != True:
            if basicstep is None:
                if self.Model=="MG94":
                     basic = np.maximum(self.x[5] / (2 * (np.maximum(np.log(abs(self.x[5]) + 1), 1))) ,
                                       0.2) / 10
                     step = nd.step_generators.MaxStepGenerator(base_step=basic)

                     H = nd.Hessian(self.objective_wo_derivative1,step=step)(np.float32((self.x[5:7])))
                else:
                    basic = np.maximum(self.x[4] / (2 * (np.maximum(np.log(abs(self.x[4]) + 1), 1))) ,
                                       0.2) / 10
                    step = nd.step_generators.MaxStepGenerator(base_step=basic)
                    H = nd.Hessian(self.objective_wo_derivative1,step=step)(np.float32((self.x[4:6])))

            else:
                if self.Model == "MG94":
                    basic = basicstep
                    step = nd.step_generators.MaxStepGenerator(base_step=basic)
                    H = nd.Hessian(self.objective_wo_derivative1, step=step)(np.float32((self.x[5:7])))
                else:
                    basic = basicstep
                    step = nd.step_generators.MaxStepGenerator(base_step=basic)
                    H = nd.Hessian(self.objective_wo_derivative1, step=step)(np.float32((self.x[4:6])))


        else:
            if self.Model == "MG94":
                basic = np.maximum(np.exp(self.x[5]) / (2 * (np.maximum(np.log(np.exp(self.x[5]) + 1), 1))) - 0.5,0.2)/10
                step = nd.step_generators.MaxStepGenerator(base_step=basic)
                H = nd.Hessian(self.objective_wo_derivative1,step=step)([np.exp(self.x[5]),self.x[6]])
            else:
                basic = np.maximum(np.exp(self.x[4]) / (2 * (np.maximum(np.log(np.exp(self.x[4]) + 1), 1))) - 0.5,0.2)/10
                step = nd.step_generators.MaxStepGenerator(base_step=basic)
                H = nd.Hessian(self.objective_wo_derivative1,step=step)([np.exp(self.x[4]),self.x[5]])

        H=np.linalg.inv(H)

        return H

#### thos two get prop for IGC
    def get_summary(self,approx=True,branchtau=False):
        scene=self.get_scene()
        ttt = len(scene['tree']["column_nodes"])
        exp_blen=scene['tree']["edge_rate_scaling_factors"]

        self.jointly_common_ancstral_inference()
        self.id = self.compute_paralog_id(ifdnalevel=False)
        igc_total = 0
        mutatation_total = 0

        if approx==False:

            ### make a dic
            list = []
            matrix_dim = len(scene['process_definitions'][0]['row_states'])
            for i in range(matrix_dim):
                list.append(tuple(np.concatenate([scene['process_definitions'][0]['row_states'][i],
                                                  scene['process_definitions'][0]['column_states'][i]])))

            index = np.arange(matrix_dim)

            dictionary = dict(zip(list, index))

            for j in range(ttt):
                    if not j == 1:
                        ini2 = self.node_to_num[self.edge_list[j][0]]
                        end2 = self.node_to_num[self.edge_list[j][1]]
                        ini1 = deepcopy(self.sites[ini2,])
                        end1 = deepcopy(self.sites[end2,])
                        result=self.whether_IGC(ini=ini1,end=end1,paralog_id=self.id[j],idnum=j,scene=scene,dic=dictionary)
                        igc_total +=result[0]
                        mutatation_total +=result[1]

            return igc_total, mutatation_total

        else:

            expect_num_igc=self._ExpectedNumGeneconv()
            expect_num_pm=self._ExpectedpointMutationNum()
            igc_total=np.sum(expect_num_igc)
            mutatation_total=np.sum(expect_num_pm)+igc_total

            if branchtau == True:

                        expected_DwellTime=self._ExpectedHetDwellTime()
                        if self.Model=="MG94":
                            taulist = [expect_num_igc[i] / (exp_blen[i] * (
                            expected_DwellTime[0][i] + self.omega * expected_DwellTime[1][i])) \
                                        if (expected_DwellTime[0][i] + self.omega *expected_DwellTime[1][i]) != 0 else 0
                                            for i in range(ttt)]
                        else:
                             taulist = [expect_num_igc[i] / (exp_blen[i] *
                                             expected_DwellTime[i]) if expected_DwellTime[i] != 0 else 0
                                        for i in range(ttt)]


                        return igc_total, mutatation_total,taulist

            else:

                return igc_total, mutatation_total




    def redesign_HKYGeneconv(self):
        # print ('tau = ', self.tau)
        Qbasic = self.get_HKYBasic()
        rate_geneconv = []

        bstau=np.exp(self.branch_tau)


        for i, pair_from in enumerate(product('ACGT', repeat=2)):
            na, nb = pair_from
            for j, pair_to in enumerate(product('ACGT', repeat=2)):
                nc, nd = pair_to
                if i == j:
                    continue
                GeneconvRate = get_HKYGeneconvRate(pair_from, pair_to, Qbasic, bstau)
                if GeneconvRate != 0.0:
                    rate_geneconv.append(GeneconvRate)
                if na == nb and nc == nd:
                    rate_geneconv.append(GeneconvRate)



        return  rate_geneconv


    def redesign__MG94Geneconv(self):
        Qbasic = self.get_MG94Basic()
        rate_geneconv = []
        bstau = np.exp(self.branch_tau)


        for i, pair in enumerate(product(self.codon_nonstop, repeat=2)):
            # use ca, cb, cc to denote codon_a, codon_b, codon_c, where cc != ca, cc != cb
            ca, cb = pair
            sa = self.codon_to_state[ca]
            sb = self.codon_to_state[cb]
            if ca != cb:
                for cc in self.codon_nonstop:
                    if cc == ca or cc == cb:
                        continue
                    sc = self.codon_to_state[cc]
                    # (ca, cb) to (ca, cc)
                    Qb = Qbasic[sb, sc]
                    if Qb != 0:
                        rate_geneconv.append(Qb)

                    # (ca, cb) to (cc, cb)
                    Qb = Qbasic[sa, sc]
                    if Qb != 0:
                        rate_geneconv.append(Qb)


                # (ca, cb) to (ca, ca)
                Qb = Qbasic[sb, sa]
                if isNonsynonymous(cb, ca, self.codon_table):
                    Tgeneconv = bstau * self.omega
                else:
                    Tgeneconv = bstau
                rate_geneconv.append(Qb + Tgeneconv)

                # (ca, cb) to (cb, cb)

                Qb = Qbasic[sa, sb]
                rate_geneconv.append(Qb + Tgeneconv)


            else:
                for cc in self.codon_nonstop:
                    if cc == ca:
                        continue
                    sc = self.codon_to_state[cc]
                    Qb = Qbasic[sa, sc]
                    if Qb != 0:
                        # (ca, ca) to (ca,  cc)
                        rate_geneconv.append(Qb)
                        # (ca, ca) to (cc, ca)
                        rate_geneconv.append(Qb)
                        # (ca, ca) to (cc, cc)
                        rate_geneconv.append(0.0)


        return rate_geneconv


    def making_Qmatrix(self):

        if self.Model=="HKY":
           Q_trans=self.redesign_HKYGeneconv()

        else:
           Q_trans = self.redesign__MG94Geneconv()

        scene = self.get_scene()

        actual_number = len(Q_trans)

        global x_i
        x_i = 0

        global index
        index = 0

        # dic is from 1:16

        if self.Model == 'HKY':
            self.Q = np.zeros(shape=(16, 9))
            self.dic_col = np.zeros(shape=(16, 9))
            for i in range(actual_number):

                # x_io means current index for row states, x_i is states for last times
                # self.dic_col indicates the coordinates for ending states

                x_io = (scene['process_definitions'][1]['row_states'][i][0]) * 4 + (scene['process_definitions'][1][
                    'row_states'][i][1])

                if x_i == x_io:
                    self.Q[x_io, index] = Q_trans[i]
                    self.dic_col[x_io, index] = 1 + (scene['process_definitions'][1]['column_states'][i][0]) * 4 + (
                        scene['process_definitions'][1]['column_states'][i][1])
                    x_i = x_io
                    index = index + 1

                else:
                    self.Q[x_io, 0] = Q_trans[i]
                    self.dic_col[x_io, 0] = 1 + (scene['process_definitions'][1]['column_states'][i][0]) * 4 + (
                        scene['process_definitions'][1]['column_states'][i][1])
                    x_i = x_io
                    index = 1

        else:
            self.Q = np.zeros(shape=(3721, 27))
            self.dic_col = np.zeros(shape=(3721, 27))
            for i in range(actual_number):
                x_io = (scene['process_definitions'][1]['row_states'][i][0]) * 61 + (scene[
                    'process_definitions'][1]['row_states'][i][1])
                if x_i == x_io:
                    self.Q[x_io, index] = Q_trans[i]
                    self.dic_col[x_io, index] = 1 + (scene['process_definitions'][1]['column_states'][i][0]) * 61 + (
                        scene['process_definitions'][1]['column_states'][i][1])
                    x_i = x_io
                    index = index + 1


                else:
                    self.Q[x_io, 0] = Q_trans[i]
                    self.dic_col[x_io, 0] = 1 + (scene['process_definitions'][1]['column_states'][i][0]) * 61 + (
                        scene['process_definitions'][1]['column_states'][i][1])
                    x_i = x_io
                    index = 1


        return self.Q, self.dic_col

    def original_Q(self,if_rebuild_Q=True):
        if if_rebuild_Q==True:
           self.making_Qmatrix()

        if self.Model == 'HKY':
            self.Q_original = np.zeros(shape=(16, 16))
            for i in range(16):
                for j in range(9):
                    index=int(self.dic_col[i,j]-1)
                    if(index>=0):
                        self.Q_original[i,index]=self.Q[i,j]

            for k in  range(16):
                self.Q_original[k,k]=-sum(self.Q_original[k,])

        else:
            self.Q_original = np.zeros(shape=(3721, 3721))
            for i in range(3721):
                for j in range(27):
                    index=int(self.dic_col[i,j]-1)
                    self.Q_original[i,index]=self.Q[i,j]

            for k in  range(3721):
                self.Q_original[k, k]=-sum(self.Q_original[k,])


    def loglikelihood_branch(self):

            self.original_Q(if_rebuild_Q=True)
            Q = linalg.expm(self.Q_original * self.time)

            logllsum=0

            for ll in range(self.nsites):
                log_value=Q[int(self.ini[ll])][int(self.end[ll])]
                if log_value>0:
                    logllsum +=np.log(log_value)

            return logllsum

    def loglikelihood_branch_gradient(self, x_new):
        '''
        Modified from Alex's objective_and_gradient function in ctmcaas/adv-log-likelihoods/mle_geneconv_common.py
        '''

        self.branch_tau = deepcopy(x_new)
        x = deepcopy(x_new)

        ll=self.loglikelihood_branch()



        delta = deepcopy(max(1, abs((self.branch_tau))) * 0.000001)

        branchtau=deepcopy((self.branch_tau))

        x_plus_delta1 = deepcopy(branchtau)
        x_plus_delta1 += delta / 2
        self.branch_tau=(x_plus_delta1)
        ll_delta1 = self.loglikelihood_branch()

            # ll_delta0 is f(x-h/2)
        x_plus_delta0 = deepcopy(branchtau)
        x_plus_delta0 -= delta/2
        self.branch_tau = (x_plus_delta0)
        ll_delta0 = self.loglikelihood_branch()

        d_estimate = (ll_delta1 - ll_delta0) / delta

        self.branch_tau = x

        print('log likelihood = ', ll, flush=True)
        print(' derivatives = ', d_estimate, flush=True)
        print('tau:',self.branch_tau, flush=True)

        f = -ll
        g = -d_estimate

        return f, g

    def get_branch_mle(self,branch):


        guess_x=np.log(np.power(self.id[branch],self.K)*self.tau)
        self.branch_tau=deepcopy(guess_x)
        bnds = [(-10.0,5.0)]

        result = scipy.optimize.minimize(self.loglikelihood_branch_gradient, guess_x, jac=True, method='L-BFGS-B',bounds=bnds,
                                         options={'gtol': 1e-07,
                                                  'eps': 1e-09,
                                                  'maxls': 25 })
        print(result)


        return np.exp(self.branch_tau)


    def test(self):
        self.get_mle()
        self.jointly_common_ancstral_inference()
        self.id = self.compute_paralog_id(ifdnalevel=False)



        ttt = len(self.tree['col'])

        for j in range(ttt):
            if self.Model == "HKY":
                if  j > 1:
                    ini2 = self.node_to_num[self.edge_list[j][0]]
                    end2 = self.node_to_num[self.edge_list[j][1]]
                    self.ini = deepcopy(self.sites[ini2,])
                    self.end = deepcopy(self.sites[end2,])
                    self.time=np.exp(self.x_rates[j])
                    print(self.get_branch_mle(branch=j))

            if self.Model == "MG94":
                if  j > 1:
                    ini2 = self.node_to_num[self.edge_list[j][0]]
                    end2 = self.node_to_num[self.edge_list[j][1]]
                    self.ini = deepcopy(self.sites[ini2,])
                    self.end = deepcopy(self.sites[end2,])
                    self.time = np.exp(self.x_rates[j])
                    print(self.get_branch_mle(branch=j))

    def sum_branch(self,MAX=4,epis=0.05,K=None):

        list = []
        if self.only_hessian==False:


            list.append(self.nsites)
            ll0 = self.get_mle()["fun"]
            list.append(ll0)
            list.append(self.tau)
            if self.Model == "MG94":
                list.append(self.omega)
            pstau = deepcopy(self.tau)

    ###### oldid
            self.id = self.compute_paralog_id()
            idold = deepcopy(self.id)
            print(self.id)

            for j in range(len(self.edge_list)):
                list.append(idold[j])
            for j in range(len(self.edge_list)):
                list.append(self.x_rates[j])

            ttt = len(self.tree['col'])
            for j in range(ttt):
                if self.Model == "HKY":
                    if j > 1:
                        ini2 = self.node_to_num[self.edge_list[j][0]]
                        end2 = self.node_to_num[self.edge_list[j][1]]
                        self.ini = deepcopy(self.sites[ini2,])
                        self.end = deepcopy(self.sites[end2,])
                        self.time = np.exp(self.x_rates[j])
                     #   bstau=self.get_branch_mle(branch=j)[0]
                        #    print(bstau)
                    #    list.append(bstau)


                if self.Model == "MG94":
                    if j > 1:
                        ini2 = self.node_to_num[self.edge_list[j][0]]
                        end2 = self.node_to_num[self.edge_list[j][1]]
                        self.ini = deepcopy(self.sites[ini2,])
                        self.end = deepcopy(self.sites[end2,])
                        self.time = np.exp(self.x_rates[j])
                        #    bstau = self.get_branch_mle(branch=j)[0]
                        #     print(bstau)
                        #    list.append(bstau)


            self.ifmodel = "EM_full"
            if K is not None:
                self.K=K
            self.get_initial_x_process()
            pstau = deepcopy(self.tau)

            if self.if_rerun==True:

                self.get_mle()
                ll1 = self.get_mle()["fun"]

                self.update_by_x(self.x)
                self.id = self.compute_paralog_id()
                self.update_by_x(self.x)
                self.id = self.compute_paralog_id()

                difference = abs(self.tau - pstau)
                print(difference)

                print("EMcycle:")
                print(0)
                print(self.id)
                print(self.K)
                print(self.tau)
                print("xxxxxxxxxxxxxxxxx")
                print("xxxxxxxxxxxxxxxxx")
                print("xxxxxxxxxxxxxxxxx")
                print("\n")

                i = 1
                while i <= MAX and difference >= epis:
                    pstau_1 = deepcopy(self.tau)

                    ll1 = self.get_mle()["fun"]
                    self.update_by_x(self.x)
                    self.id = self.compute_paralog_id()
                    self.update_by_x(self.x)
                    self.id = self.compute_paralog_id()
                    difference = abs(self.tau - pstau_1)
                    print(difference)

                    print("EMcycle:")
                    print(i)
                    i = i + 1
                    print(self.id)
                    print("K:")
                    print(self.K)
                    print("Tau:")
                    print(self.tau)
                    print("xxxxxxxxxxxxxxxxx")
                    print("xxxxxxxxxxxxxxxxx")
                    print("xxxxxxxxxxxxxxxxx")
                    print("\n")


                print("new tau: ", self.tau)
                print("K: ", self.K)
                print("new ll: ", ll1)
                list.append(ll1)
                if self.Model == "MG94":
                    list.append(self.omega)
                list.append(self.tau)
                list.append(self.K)

                for j in range(len(self.edge_list)):
                    list.append(self.id[j])
                for j in range(len(self.edge_list)):
                    list.append(self.x_rates[j])

                hessian = self.get_Hessian()
                print(hessian)
                list.append(hessian[0][0])
                list.append(hessian[1][1])
                list.append(hessian[1][0])
                self.ifexp = True
                hessian = self.get_Hessian()
                print(hessian)
                list.append(hessian[0][0])
                list.append(hessian[1][1])
                list.append(hessian[1][0])

                self.id=self.compute_paralog_id()
                ttt = len(self.tree['col'])
                for j in range(ttt):
                    if self.Model == "HKY":
                        if j > 1:
                            ini2 = self.node_to_num[self.edge_list[j][0]]
                            end2 = self.node_to_num[self.edge_list[j][1]]
                            self.ini = deepcopy(self.sites[ini2,])
                            self.end = deepcopy(self.sites[end2,])
                            self.time = np.exp(self.x_rates[j])
                            bstau = self.get_branch_mle(branch=j)[0]
                            print(bstau)
                            list.append(bstau)

                    if self.Model == "MG94":
                        if j > 1:
                            ini2 = self.node_to_num[self.edge_list[j][0]]
                            end2 = self.node_to_num[self.edge_list[j][1]]
                            self.ini = deepcopy(self.sites[ini2,])
                            self.end = deepcopy(self.sites[end2,])
                            self.time = np.exp(self.x_rates[j])
                            bstau = self.get_branch_mle(branch=j)[0]
                            print(bstau)
                            list.append(bstau)

            else:
                self.id = self.compute_paralog_id()
                self.ifexp = True
                hessian = self.get_Hessian()
                print(hessian)
                list.append(hessian[0][0])
                list.append(hessian[1][1])

                for j in range(len(self.edge_list)):
                    list.append(self.id[j])
                for j in range(len(self.edge_list)):
                    list.append(self.x_rates[j])

                for j in range(len(self.edge_list)):
                    if self.Model == "HKY":
                        if j > 1:
                            ini2 = self.node_to_num[self.edge_list[j][0]]
                            end2 = self.node_to_num[self.edge_list[j][1]]
                            self.ini = deepcopy(self.sites[ini2,])
                            self.end = deepcopy(self.sites[end2,])
                            self.time = np.exp(self.x_rates[j])
                       #     bstau = self.get_branch_mle(branch=j)[0]
                         #   list.append(bstau)

                    if self.Model == "MG94":
                        if j > 1:
                            ini2 = self.node_to_num[self.edge_list[j][0]]
                            end2 = self.node_to_num[self.edge_list[j][1]]
                            self.ini = deepcopy(self.sites[ini2,])
                            self.end = deepcopy(self.sites[end2,])
                            self.time = np.exp(self.x_rates[j])
                        #    bstau = self.get_branch_mle(branch=j)[0]
                        #    list.append(bstau)


            print(list)
            save_nameP = self.save_path + self.Model + "new_tau" + '.txt'


        else:

            print(self.x)
            self.id = self.compute_paralog_id()
            print(self.id, flush=True)

            self.ifmodel = "EM_full"
            self.get_initial_x_process()
            self.ini_by_file()
            print(self.tau,flush=True)
            print(self.id,flush=True)

            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            step=[0.01,0.05,0.1,0.2,0.3,0.5,0.8,1,2,3,5]

            self.ifexp = False

            for i in range(10):
                hessian=self.get_Hessian(basicstep=step[i])
                print(hessian,flush=True)
                list.append(hessian[0][0])
                list.append(hessian[1][1])
                list.append(hessian[1][0])

            save_nameP = self.save_path + self.Model + "new_tau_noh" + '.txt'

        with open(save_nameP, 'wb') as f:
            np.savetxt(f, list)

    def measure_difference_two(self,a,b):
        index=0

        for i in range(3):
            if a[i]!=b[i]:
                index=index+1

        return index

    def _ExpectedHetDwellTime(self, package='new', display=False):

        if package == 'new':
            self.scene_ll = self.get_scene()
            if self.Model == 'MG94':
                syn_heterogeneous_states = [(a, b) for (a, b) in
                                            list(product(range(len(self.codon_to_state)), repeat=2)) if
                                            a != b and self.isSynonymous(self.codon_nonstop[a], self.codon_nonstop[b])]
                nonsyn_heterogeneous_states = [(a, b) for (a, b) in
                                               list(product(range(len(self.codon_to_state)), repeat=2)) if
                                               a != b and not self.isSynonymous(self.codon_nonstop[a],
                                                                                self.codon_nonstop[b])]
                dwell_request = [dict(
                    property='SDWDWEL',
                    state_reduction=dict(
                        states=syn_heterogeneous_states,
                        weights=[2] * len(syn_heterogeneous_states)
                    )),
                    dict(
                        property='SDWDWEL',
                        state_reduction=dict(
                            states=nonsyn_heterogeneous_states,
                            weights=[2] * len(nonsyn_heterogeneous_states)
                        ))
                ]

            elif self.Model == 'HKY':
                heterogeneous_states = [(a, b) for (a, b) in list(product(range(len(self.nt_to_state)), repeat=2)) if
                                        a != b]
                dwell_request = [dict(
                    property='SDWDWEL',
                    state_reduction=dict(
                        states=heterogeneous_states,
                        weights=[2] * len(heterogeneous_states)
                    )
                )]

            j_in = {
                'scene': self.scene_ll,
                'requests': dwell_request,
            }
            j_out = jsonctmctree.interface.process_json_in(j_in)

            ttt=len(self.edge_list)
            if self.Model=="MG94":
                ExpectedDwellTime=np.zeros((2,ttt))

                for i in range(len(self.edge_list)):
                    for j in range(2):
                        ExpectedDwellTime[j][i]=j_out['responses'][j][i]
            else:
                ExpectedDwellTime = np.zeros(ttt)

                for i in range(len(self.edge_list)):
                     ExpectedDwellTime[i] = j_out['responses'][0][i]


            return ExpectedDwellTime
        else:
            print('Need to implement this for old package')



    def _ExpectedHetDwellTime_DNA(self, package='new', display=False):

        if package == 'new':
            self.scene_ll = self.get_scene()
            if self.Model == 'MG94':
                syn_heterogeneous_states_1 = [(a, b) for (a, b) in
                                            list(product(range(len(self.codon_to_state)), repeat=2)) if
                                            a != b and self.isSynonymous(self.codon_nonstop[a], self.codon_nonstop[b]) and
                                            self.measure_difference_two(self.codon_nonstop[a],
                                                                                self.codon_nonstop[b])==1]
                nonsyn_heterogeneous_states_1 = [(a, b) for (a, b) in
                                               list(product(range(len(self.codon_to_state)), repeat=2)) if
                                               a != b and not self.isSynonymous(self.codon_nonstop[a],
                                                                                self.codon_nonstop[b]) and
                                               self.measure_difference_two(self.codon_nonstop[a],
                                                                                self.codon_nonstop[b])==1]
                syn_heterogeneous_states_2 = [(a, b) for (a, b) in
                                              list(product(range(len(self.codon_to_state)), repeat=2)) if
                                              a != b and self.isSynonymous(self.codon_nonstop[a],
                                                                           self.codon_nonstop[b]) and
                                              self.measure_difference_two(self.codon_nonstop[a],
                                                                          self.codon_nonstop[b]) == 2]
                nonsyn_heterogeneous_states_2 = [(a, b) for (a, b) in
                                                 list(product(range(len(self.codon_to_state)), repeat=2)) if
                                                 a != b and not self.isSynonymous(self.codon_nonstop[a],
                                                                                  self.codon_nonstop[b]) and
                                                 self.measure_difference_two(self.codon_nonstop[a],
                                                                             self.codon_nonstop[b]) == 2]
                syn_heterogeneous_states_3 = [(a, b) for (a, b) in
                                              list(product(range(len(self.codon_to_state)), repeat=2)) if
                                              a != b and self.isSynonymous(self.codon_nonstop[a],
                                                                           self.codon_nonstop[b]) and
                                              self.measure_difference_two(self.codon_nonstop[a],
                                                                          self.codon_nonstop[b]) == 3]
                nonsyn_heterogeneous_states_3 = [(a, b) for (a, b) in
                                                 list(product(range(len(self.codon_to_state)), repeat=2)) if
                                                 a != b and not self.isSynonymous(self.codon_nonstop[a],
                                                                                  self.codon_nonstop[b]) and
                                                 self.measure_difference_two(self.codon_nonstop[a],
                                                                             self.codon_nonstop[b]) == 3]
                dwell_request = [dict(
                    property='SDWDWEL',
                    state_reduction=dict(
                        states=syn_heterogeneous_states_1,
                        weights=[2] * len(syn_heterogeneous_states_1)
                    )),
                    dict(
                        property='SDWDWEL',
                        state_reduction=dict(
                            states=nonsyn_heterogeneous_states_1,
                            weights=[2] * len(nonsyn_heterogeneous_states_1)
                        )),
                    dict(
                        property='SDWDWEL',
                        state_reduction=dict(
                            states=syn_heterogeneous_states_2,
                            weights=[2] * len(syn_heterogeneous_states_2)
                        )),
                    dict(
                        property='SDWDWEL',
                        state_reduction=dict(
                            states=nonsyn_heterogeneous_states_2,
                            weights=[2] * len(nonsyn_heterogeneous_states_2)
                        )),
                    dict(
                        property='SDWDWEL',
                        state_reduction=dict(
                            states=syn_heterogeneous_states_3,
                            weights=[2] * len(syn_heterogeneous_states_3)
                        )),
                    dict(
                        property='SDWDWEL',
                        state_reduction=dict(
                            states=nonsyn_heterogeneous_states_3,
                            weights=[2] * len(nonsyn_heterogeneous_states_3)
                        ))
                ]

            elif self.Model == 'HKY':
                heterogeneous_states = [(a, b) for (a, b) in list(product(range(len(self.nt_to_state)), repeat=2)) if
                                        a != b]
                dwell_request = [dict(
                    property='SDWDWEL',
                    state_reduction=dict(
                        states=heterogeneous_states,
                        weights= [2] * len(heterogeneous_states)
                    )
                )]

            j_in = {
                'scene': self.scene_ll,
                'requests': dwell_request,
            }
            j_out = jsonctmctree.interface.process_json_in(j_in)

            ttt=len(self.edge_list)
            if self.Model=="MG94":
                ExpectedDwellTime=np.zeros((6,ttt))

                for i in range(ttt):
                    for j in range(6):
                        ExpectedDwellTime[j][i]=j_out['responses'][j][i]
            else:
                ExpectedDwellTime = np.zeros(ttt)

                for i in range(len(self.edge_list)):
                     ExpectedDwellTime[i] = j_out['responses'][0][i]



            return ExpectedDwellTime
        else:
            print('Need to implement this for old package')


# Generate parameter for simulation

    def read_parameter_gls(self,if_rerun=False,ifmodel="EM_full"):

        if ifmodel=="EM_full" and if_rerun==False:

            ttt = len(self.tree['col'])
            id = np.ones(ttt)
            self.id=id
            self.ifmodel = "EM_full"
            self.get_initial_x_process()
            self.get_processes()

            scene=self.get_scene()

        else:
            self.get_initial_x_process()
            self.get_processes()
            scene = self.get_scene()

        return scene

    def sum_branch_test(self ,id0=None):

            self.get_mle()
            self.id = self.compute_paralog_id()
            self.ifmodel = "EM_full"

            print("xxxxxxxxxxxxxxx")
            self.get_initial_x_process()

            if id0==None:
               self.id = self.compute_paralog_id()
            else:
                self.id =id0


            self.ifexp = False
            hessian = self.get_Hessian1()
            list=[]
            print(hessian)
            list.append(hessian[0][0])
            list.append(hessian[1][1])

            self.ifexp = True
            hessian = self.get_Hessian1()
            list=[]
            print(hessian)
            list.append(hessian[0][0])
            list.append(hessian[1][1])

            self.ifexp = False
            hessian = self.get_Hessian1(setstep=True)
            list = []
            print(hessian)
            list.append(hessian[0][0])
            list.append(hessian[1][1])

            self.ifexp = True
            hessian = self.get_Hessian1(setstep=True)
            list = []
            print(hessian)
            list.append(hessian[0][0])
            list.append(hessian[1][1])

    def get_Hessian1(self,setstep=False):

        if setstep==False:

            if self.ifexp != True:
                if self.Model == "MG94":
                    basic = np.maximum(self.x[5] / (2 * (np.maximum(np.log(abs(self.x[5]) + 1), 1))),
                                       0.2) / 10
                    step = nd.step_generators.MaxStepGenerator(base_step=basic)

                    H = nd.Hessian(self.objective_wo_derivative1, step=step)(np.float32((self.x[5:7])))
                else:
                    basic = np.maximum(self.x[4] / (2 * (np.maximum(np.log(abs(self.x[4]) + 1), 1))),
                                       0.2) / 10
                    step = nd.step_generators.MaxStepGenerator(base_step=basic)
                    H = nd.Hessian(self.objective_wo_derivative1, step=step)(np.float32((self.x[4:6])))

            else:
                if self.Model == "MG94":
                    basic = np.maximum(np.exp(self.x[5]) / (2 * (np.maximum(np.log(np.exp(self.x[5]) + 1), 1))) - 0.5,
                                       0.2) / 10
                    step = nd.step_generators.MaxStepGenerator(base_step=basic)
                    H = nd.Hessian(self.objective_wo_derivative1, step=step)([np.exp(self.x[5]), self.x[6]])
                else:
                    basic = np.maximum(np.exp(self.x[4]) / (2 * (np.maximum(np.log(np.exp(self.x[4]) + 1), 1))) - 0.5,
                                       0.2) / 10
                    step = nd.step_generators.MaxStepGenerator(base_step=basic)
                    H = nd.Hessian(self.objective_wo_derivative1, step=step)([np.exp(self.x[4]), self.x[5]])

            H = np.linalg.inv(H)


        else:

                if self.ifexp != True:
                    if self.Model == "MG94":
                        basic = [deepcopy(max(1,abs(self.x[5]))*0.001),deepcopy(max(1,abs(self.x[6]))*0.001)]
                        step = nd.step_generators.MaxStepGenerator(base_step=basic,step_nom=1)

                        H = nd.Hessian(self.objective_wo_derivative1, step=step)(np.float32((self.x[5:7])))
                    else:
                        basic = [deepcopy(max(1,abs(self.x[4]))*0.0001),deepcopy(max(1,abs(self.x[5]))*0.01)]
                        step = nd.step_generators.MaxStepGenerator(base_step=basic,step_nom=1)
                        H = nd.Hessian(self.objective_wo_derivative1, step=step)(np.float32((self.x[4:6])))

                else:
                    if self.Model == "MG94":
                        basic = [deepcopy(max(1, np.exp(self.x[5])) * 0.001), deepcopy(max(1, abs(self.x[6])) * 0.001)]
                        step = nd.step_generators.MaxStepGenerator(base_step=basic, step_nom=1)
                        H = nd.Hessian(self.objective_wo_derivative1, step=step)([np.exp(self.x[5]), self.x[6]])
                    else:
                        basic = [deepcopy(max(1, np.exp(self.x[4])) * 0.001), deepcopy(max(1, abs(self.x[5])) * 0.001)]
                        step = nd.step_generators.MaxStepGenerator(base_step=basic, step_nom=1)
                        H = nd.Hessian(self.objective_wo_derivative1, step=step)([np.exp(self.x[4]), self.x[5]])

                H = np.linalg.inv(H)

                return H

        return H








if __name__ == '__main__':



    name = "YBL087C_YER117W_input"

    paralog = ['YBL087C', 'YER117W']
    alignment_file = '../test/yeast/' + name + '.fasta'
    newicktree = '../test/yeast/YeastTree.newick'


    Force = None
    model = 'HKY'

    save_name = model+name
    geneconv = Embrachtau1(newicktree, alignment_file, paralog, Model=model, Force=Force,
                               save_path='../test/save/', save_name=save_name,kbound=5)

  #  geneconv.get_mle()


    geneconv.sum_branch()

   # geneconv.sum_branch_test(id0=[0.9326130134925631, 1.0, 0.834609616220603, 0.8267228498499286,
                          #        0.7941712107333674, 0.7849171211462617, 0.7896771655905308, 0.758017116692181,
                          #        0.7799356954180342, 0.7597093258876664, 0.7455012840422707, 0.7677272289953266])


  #  geneconv.get_mle()
   # print(timeit(geneconv.objective_wo_derivative(geneconv.x)))
  #  print(geneconv.get_Hessian())


  #  print(geneconv.compute_paralog_id())
  #  geneconv.dwell_id=False
 #   print(geneconv.compute_paralog_id())


   # geneconv.EM_branch_tau(MAX=5,epis=0.01,force=None,K=2)






   # self.get_paralog_diverge()
  #  print(geneconv.omega)
  #  print(geneconv.IGC_Omega)
