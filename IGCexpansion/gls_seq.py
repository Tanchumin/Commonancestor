# coding=utf-8
# A separate file for G_S to simulation by sequence level

# # Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function
from em_pt1 import *
from copy import deepcopy
from CodonGeneconFunc import *
import os
import numpy as np
import scipy
import array
from numpy import random
from scipy import linalg
import copy
from scipy.stats import poisson

from IGCexpansion.CodonGeneconFunc import isNonsynonymous
import numpy.core.multiarray


class GSseq:

    def __init__(self,
                 geneconv=None,newicktree=None,
                 sizen=400,branch_list=None,K=0,fix_tau=None,
                 pi=None,omega=None,kappa=None,
                 ifmakeQ=False,save_name=None,Model="HKY",
                 save_path=None,tract_len=None,ifDNA=False,
                 ifmodel="EM_full",
                 ):

        self.geneconv                 = geneconv
        self.newicktree               = newicktree
        self.post_dup                 = "N1"
        self.save_path                =save_path
        self.save_name = save_name

        self.ancestral_state_response = None
        self.scene                    = None
        self.num_to_state             = None
        self.num_to_node              = None
        self.node_to_num              = None
        self.node_length              = None


        self.omega=omega
        self.kappa=kappa
        self.pi=pi

# change Q
        self.Q_new = None
        self.Q= None
        self.dic_col= None
        self.ifmakeQ=ifmakeQ


# model setting
        self.Model=Model
        self.sites = None


#making Q matrix
        bases = 'tcag'.upper()
        codons = [a + b + c for a in bases for b in bases for c in bases]
        amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'

        self.nt_to_state = {a: i for (i, a) in enumerate('ACGT')}
        self.state_to_nt = {i: a for (i, a) in enumerate('ACGT')}

        self.codon_table = dict(zip(codons, amino_acids))
        self.codon_nonstop = [a for a in self.codon_table.keys() if not self.codon_table[a] == '*']
        self.codon_to_state = {a.upper(): i for (i, a) in enumerate(self.codon_nonstop)}
        self.state_to_codon = {i: a.upper() for (i, a) in enumerate(self.codon_nonstop)}
        self.observable_nodes = []


## simulation setting
        self.sizen=sizen
        self.t= branch_list
        self.K=K
## self.tau is used to update for Q matriex
        self.tau =fix_tau
        self.fix_tau=fix_tau


        self.tract_len=tract_len
        self.point_mutation= None

        self.ifDNA=ifDNA


## self.listprop is to record the prop of change
        self.listprop={1:0}
        self.listproptimes = {1:0}

        # decide which model to run

        self.ifmodel=ifmodel




        self.initialize_parameters()

    def initialize_parameters(self):


        self.hash_event={}
        for i in range(50):
            self.hash_event[i] = 0


        self.hash_event_t={}
        for i in range(50):
            self.hash_event_t[i] = 0




        if self.ifmakeQ==False:
            print("The parameters and tree are inherited from inputs")
            self.tree = deepcopy(self.geneconv.tree)
            self.scene=self.geneconv.read_parameter_gls(ifmodel=self.ifmodel)
            self.pi = deepcopy(self.geneconv.pi)
            self.t = self.geneconv.tree['rate']
            self.kappa=self.geneconv.kappa
            self.omega=self.geneconv.omega
            self.tau=self.geneconv.tau
            self.fix_tau=self.geneconv.tau
            if self.ifmodel!="old":
                  self.K=self.geneconv.K
            self.observable_nodes=self.geneconv.observable_nodes
            self.num_to_node = self.geneconv.num_to_node


        else:
            # a,b,c is useless
            self.tree, b,self.node_to_num,length = read_newick(self.newicktree, self.post_dup)
            self.num_to_node = {self.node_to_num[i]: i for i in self.node_to_num}
            self.t=length
            for i in range(len(length)):
                if (i+1) not in set(self.tree['row']):
                    self.observable_nodes.append(i+1)





    def get_MG94BasicRate(self,ca, cb, pi, kappa, omega, codon_table):
        dif = [ii for ii in range(3) if ca[ii] != cb[ii]]
        ndiff = len(dif)
        if ndiff > 1:
            return 0
        elif ndiff == 0:
            print('Please check your codon tables and make sure no redundancy')
            return 0
        else:
            na = ca[dif[0]]
            nb = cb[dif[0]]
            QbasicRate = pi['ACGT'.index(nb)]

            if self.isTransition(na, nb):
                QbasicRate *= kappa

            if self.isNonsynonymous(ca, cb, codon_table):
                QbasicRate *= omega

            return QbasicRate


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
                    Tgeneconv = self.tau * self.omega
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
                Qbasic[self.codon_to_state[ca], self.codon_to_state[cb]] = self.get_MG94BasicRate(ca, cb, self.pi,
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
   #     print(rate_geneconv)
        # process_basic is for HKY_Basic which is equivalent to 4by4 rate matrix
        return [process_basic, process_geneconv]


    def isNonsynonymous(self, ca, cb, codon_table):
        return (codon_table[ca] != codon_table[cb])


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

    def isTransition(self,na, nb):
       return (set([na, nb]) == set(['A', 'G']) or set([na, nb]) == set(['C', 'T']))




    def get_station_dis(self):
        pi=np.zeros(61)
        for ca in self.codon_nonstop:
            p=1
            for i in range(3):
                p=p*self.pi['ACGT'.index(ca[i])]
            pi[int(self.codon_to_state[ca])]=p

        pi=pi/sum(pi)

        return(pi)

    def make_ini(self):
            ini = np.ones(self.sizen)

            if self.Model=="HKY":
                z = self.pi
                sample=np.ones(4)
                for i in range(16):
                    if(i // 4 == i%4):
                        sample[i%4]=i
                for i in range(self.sizen):
                    ini[i] = int(np.random.choice(sample, 1, p=(z))[0])

            else:
                z=self.get_station_dis()
                sample = np.ones(61)
                for i in range(3721):
                    if (i // 61 == i % 61):
                        sample[i % 61] = i
                for i in range(self.sizen):
                    ini[i] = int(np.random.choice(sample, 1, p=(z))[0])

            return (ini)


 ###   supporting function

    def making_Qg(self):

        if self.Q is None:
            self.making_Qmatrix()

        global di
        global di1

        if self.Model == "HKY":
            di = 16
            di1 = 9
            self.Q_new = np.zeros(shape=(di, di1))

        else:
            di = 3721
            di1 = 27
            self.Q_new = np.zeros(shape=(di, di1))

        Q_iiii = np.ones((di))
        for ii in range(di):
            Q_iiii[ii] = sum(self.Q[ii,])

        for d in range(di):
            self.Q_new[d,] = self.Q[d,] / Q_iiii[d]

        return self.Q_new

### OBTAIN FULL Q  matrix

    def making_Qmatrix(self):

        self.get_prior()


        if  self.Model=="MG94":
                    self.processes = self.get_MG94Geneconv_and_MG94()
        else:
                    self.processes = self.get_HKYGeneconv()

        process_definitions = [{'row_states': i['row'], 'column_states': i['col'], 'transition_rates': i['rate']}
                                       for i in self.processes]

        if self.ifmakeQ==True:
             self.scene = dict(
                process_definitions = process_definitions
                )


        scene=self.scene




        actual_number = (len(scene['process_definitions'][1]['transition_rates']))

        self.actual_number = actual_number

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
                    self.Q[x_io, index] = scene['process_definitions'][1]['transition_rates'][i]
                    self.dic_col[x_io, index] = 1 + (scene['process_definitions'][1]['column_states'][i][0]) * 4 + (
                        scene['process_definitions'][1]['column_states'][i][1])
                    x_i = x_io
                    index = index + 1

                else:
                    self.Q[x_io, 0] = scene['process_definitions'][1]['transition_rates'][i]
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
                    self.Q[x_io, index] = scene['process_definitions'][1]['transition_rates'][i]
                    self.dic_col[x_io, index] = 1 + (scene['process_definitions'][1]['column_states'][i][0]) * 61 + (
                        scene['process_definitions'][1]['column_states'][i][1])
                    x_i = x_io
                    index = index + 1


                else:
                    self.Q[x_io, 0] = scene['process_definitions'][1]['transition_rates'][i]
                    self.dic_col[x_io, 0] = 1 + (scene['process_definitions'][1]['column_states'][i][0]) * 61 + (
                        scene['process_definitions'][1]['column_states'][i][1])
                    x_i = x_io
                    index = 1

        return self.Q, self.dic_col

    # test stat for simulation
    def solo_difference(self, ini):
            index = 0


            if self.Model == "HKY":
                str = {0, 5, 10, 15}

                for i in range(self.sizen):
                    if not ini[i] in str:
                        index = index + 1
                index = 1 - (index / self.sizen)
            else:
                if self.ifDNA==False:
                    for i in range(self.sizen):
                        ca = (ini[i]) // 61
                        cb = (ini[i]) % 61
                        if ca != cb:
                            index = index + 1
                    index = 1 - (index / self.sizen)

                else:
                    for i in range(self.sizen):
                        ca = (ini[i]) // 61
                        cb = (ini[i]) % 61

                        cb1 = self.state_to_codon[cb]
                        ca1 = self.state_to_codon[ca]
                        for ii in range(3):
                            if cb1[ii] != ca1[ii]:
                                index = index + 1

                    index = 1 - (index / (3*self.sizen))

            return index


    def isNonsynonymous(self, ca, cb, codon_table):
        return (codon_table[ca] != codon_table[cb])


    def point_IGC(self,pre,post):
        if self.Model=="MG94":
            site_number=61
        else:
            site_number = 4


        i_b = pre // site_number
        j_b = pre % site_number
        i_p = post // site_number
        j_p = post % site_number


        igc=0
        point=0
        change_i=0
        change_j=0

        if i_p == j_p:
            if i_b != j_b and i_b == i_p:
                # y_coor is corresponding coor for igc
                y_coor = np.argwhere(self.dic_col[int(pre),] == (int(post) + 1))[0]
                qq = self.Q[int(pre), y_coor]


                cb1=self.state_to_codon[j_b]
                ca1 = self.state_to_codon[j_p]



                if self.isNonsynonymous(cb1, ca1, self.codon_table):
                    igc = (self.tau*self.omega) / qq
                else:
                    igc = (self.tau ) / qq

                point = 1 - igc

                if point < 0:
                    print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")



            elif (i_b != j_b and j_b == j_p):
                y_coor = np.argwhere(self.dic_col[int(pre),] == (int(post) + 1))[0]
                qq = self.Q[int(pre), y_coor]

                cb1 = self.state_to_codon[i_b]
                ca1 = self.state_to_codon[i_p]

                if self.isNonsynonymous(cb1, ca1, self.codon_table):
                    igc = (self.tau*self.omega) / qq
                else:
                    igc = (self.tau ) / qq

                point = 1 - igc


                if point < 0:
                    print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")



        else:
            point=1

        if i_p!=i_b:
            change_i=1
        else:
            change_j = 1



        return point,igc,change_i,change_j


# GLS algorithm on sequence level, which just generate on ini and end sequence

    def GLS_sequnce(self, t=0.1, ini=None,k=1.1, tau=1.1,iffirst=False):

        global di
        global di1


        inirel=deepcopy(ini)

        if self.Model == "HKY":
            di = 16
            di1 = 9

        else:
            di = 3721
            di1 = 27

        u = 0
        ifcontinue=True

        igc=0
        point=0
        change_i=0
        change_j=0

        while(ifcontinue==True):
            id = self.solo_difference(ini)
            self.making_Qg()
            self.change_t_Q(tau=(np.power(id, k) * tau))

            Q_iiii = np.ones((di))
            for ii in range(di):
                Q_iiii[ii] = sum(self.Q[ii,])


            p = np.zeros(self.sizen)
            lambda_change = 0
            for ll in range(self.sizen):
                lambda_change = Q_iiii[int(ini[ll])] + lambda_change
                p[ll] = Q_iiii[int(ini[ll])]


            u = u + random.exponential(1/lambda_change)
            if (u <= t):
               # print(u)

                change_location = np.random.choice(range(self.sizen), 1, p=(p/lambda_change))[0]
                change_site = int(ini[change_location])

                a = np.random.choice(range(di1), 1, p=(self.Q[change_site,] / Q_iiii[change_site]))[0]
                current_state = self.dic_col[change_site, a] - 1
                result=self.point_IGC(change_site,current_state)
                igc=result[1]+igc
                point=result[0]+point
                change_j=result[3]+change_j
                change_i = result[2] + change_i
                ini[change_location] = int(current_state)

                if iffirst==False:

                    if id in self.listprop.keys():
                        self.listprop[id] += result[1]
                        self.listproptimes[id] +=1

                    else:
                        self.listprop[id]= result[1]
                        self.listproptimes[id] = 1


            else:
                ifcontinue=False



        print("% branch length", t)
        print("% estimated number of point mutation number per site:", point / self.sizen)
        print("% estimated number of IGC number per site:", igc / self.sizen)
        print("% estimated number of one paralog  change per site:", change_i / self.sizen)
        print("% estimated number of the other paralog  change per site:", change_j / self.sizen)
        self.measure_difference(ini=deepcopy(inirel),end=deepcopy(ini))

        return ini

    def GLS_sequnce_tract(self, t=0.1, ini=None,k=1.1, tau=1.1,iffirst=False):

        global di
        global di1


        inirel=deepcopy(ini)

        if self.Model == "HKY":
            di = 16
            di1 = 9

        else:
            di = 3721
            di1 = 27

        u = 0
        ifcontinue=True

        igc=0
        point=0
        change_i=0
        change_j=0

        while(ifcontinue==True):
            id = self.solo_difference(ini)
            self.making_Qg()
            self.change_t_Q(tau=(np.power(id, k) * tau))


            Q_iiii = np.ones((di))
            for ii in range(di):
                Q_iiii[ii] = sum(self.Q[ii,])

     #       for d in range(di):
      #          self.Q_new[d,] = self.Q[d,] / Q_iiii[d]

            p = np.zeros(self.sizen)
            for ll in range(self.sizen):
                p[ll] = self.point_mutation[int(ini[ll])]

            point_change=random.uniform(0,1)

            p_IGC = np.ones(self.sizen) * (self.tau / self.tract_len)*2
            p_IGC[0] = self.tau*2

            lambda_change=sum(p) + sum(p_IGC)


            dwell=random.exponential(1 / lambda_change)
            u = u + dwell
            if (u <= t):
                id_igc = int((id * 100) // (2))-1


               # print(u)
                if point_change <= (sum(p)/lambda_change):

            #        print("ddddddddddddddddddddddddddd")

                    change_location = np.random.choice(range(self.sizen), 1, p=(p/sum(p)))[0]
                    change_site = int(ini[change_location])


                    a = np.random.choice(range(di1), 1, p=(self.Q[change_site,] / Q_iiii[change_site]))[0]
                    current_state = self.dic_col[change_site, a] - 1
                    result=self.point_IGC(change_site,current_state)
                    point=point+1
                    change_j=result[3]+change_j
                    change_i = result[2] + change_i
                    ini[change_location] = int(current_state)

                # generate IGC tract:
                else:

                    ini_change_location = np.random.choice(range(self.sizen), 1, p=(p_IGC / sum(p_IGC)))[0]
                    tract=random.geometric(1/self.tract_len)
                    result = self.tract_IGC(deepcopy(ini), tract,ini_change_location)
                    ini = result[0]
                    igc=igc+result[1]
                    change_j = result[3] + change_j
                    change_i = result[2] + change_i


                    self.hash_event[id_igc]=self.hash_event[id_igc]+result[1]

                self.hash_event_t[id_igc] = self.hash_event_t[id_igc] + (1-id)* dwell




            else:
                ifcontinue=False




        print("% branch length", t)
        print("% estimated number of point mutation number per site:", point / self.sizen)
        print("% estimated number of IGC number per site:", igc / self.sizen)
        print("% estimated number of one paralog  change per site:", change_i / self.sizen)
        print("% estimated number of the other paralog  change per site:", change_j / self.sizen)
        self.measure_difference(ini=deepcopy(inirel),end=deepcopy(ini))

        return ini

    def tract_IGC(self,ini, tract, ini_change_location):
        u=random.uniform(0,1)
        IGC=0
        num_jb=0
        num_ib=0

        if self.Model=="MG94":
            site_number=61
        else:
            site_number = 4

        if self.sizen<(ini_change_location+tract):
            tract=self.sizen-ini_change_location

        for index in range(tract):
                i=index+ini_change_location
                i_b = ini[i] // site_number
                j_b = ini[i] % site_number
                if u<=0.5:
                    new_ini=i_b*site_number+i_b
                    ini[i]=new_ini
                    if i_b !=j_b:
                        IGC=IGC+1
                        num_jb = num_jb + 1

                else:
                    new_ini=j_b*site_number+j_b
                    ini[i]=new_ini
                    if i_b !=j_b:
                        IGC=IGC+1
                        num_ib = num_ib + 1



        return ini,IGC,num_ib,num_jb





    def remake_matrix(self):
            if self.Model == "HKY":
                Q = self.get_HKYBasic()
             #   print(Q)

            if self.Model == "MG94":
                Q = self.get_MG94Basic()

            return Q

# use change_t_Q before topo so  that can make new Q conditioned on (tau^id)*K at each GLS step
    def change_t_Q(self, tau=0.99):


        if self.point_mutation is None and self.tract_len is not None:

                if self.Q is None:
                    self.making_Qmatrix()

    #sum_tau is summing up of tau to tell whether the change is IGC


                if self.Model == "HKY":
                    self.point_mutation =np.ones((16))
                    for ii in range(16):
                        sum_tau = 0
                        for jj in range(9):
                            i_b = ii // 4
                            j_b = ii % 4
                            i_p = (self.dic_col[ii, jj] - 1) // 4
                            j_p = (self.dic_col[ii, jj] - 1) % 4
                            if i_p == j_p:
                                if i_b != j_b and i_b == i_p:
                                    self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau
                                    sum_tau=sum_tau+tau
                                elif (i_b != j_b and j_b == j_p):
                                    self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau
                                    sum_tau = sum_tau + tau

                        self.point_mutation[ii] = sum(self.Q[ii,]) - sum_tau


                else:
                    self.point_mutation = np.ones((3721))
                    for ii in range(3721):
                        sum_tau=0
                        for jj in range(27):
                            i_b = ii // 61
                            j_b = ii % 61
                            i_p = (self.dic_col[ii, jj] - 1) // 61
                            j_p = (self.dic_col[ii, jj] - 1) % 61
                            if i_p == j_p:
                                if i_b != j_b and i_b == i_p:
                                    cb1 = self.state_to_codon[j_b]
                                    ca1 = self.state_to_codon[i_b]
                                    if self.isNonsynonymous(cb1, ca1, self.codon_table):
                                        self.Q[ii, jj] = self.Q[ii, jj] - (self.tau * self.omega) + (tau * self.omega)

                                        sum_tau = sum_tau + (tau * self.omega)
                                    else:
                                        self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau

                                        sum_tau = sum_tau + tau
                                elif (i_b != j_b and j_b == j_p):
                                    cb1 = self.state_to_codon[j_b]
                                    ca1 = self.state_to_codon[i_b]
                                    if self.isNonsynonymous(cb1, ca1, self.codon_table):
                                        self.Q[ii, jj] = self.Q[ii, jj] - (self.tau * self.omega) + (tau * self.omega)
                                        sum_tau = sum_tau + (tau * self.omega)
                                    else:
                                        self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau
                                        sum_tau = sum_tau + tau

                        self.point_mutation[ii]=sum(self.Q[ii, ])-sum_tau



                self.tau = tau

        else:

            if self.Q is None:
                self.making_Qmatrix()

            if self.Model == "HKY":
                for ii in range(16):
                    for jj in range(9):
                        i_b = ii // 4
                        j_b = ii % 4
                        i_p = (self.dic_col[ii, jj] - 1) // 4
                        j_p = (self.dic_col[ii, jj] - 1) % 4
                        if i_p == j_p:
                            if i_b != j_b and i_b == i_p:
                                self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau
                            elif (i_b != j_b and j_b == j_p):
                                self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau
            else:
                for ii in range(3721):
                    for jj in range(27):
                        i_b = ii // 61
                        j_b = ii % 61
                        i_p = (self.dic_col[ii, jj] - 1) // 61
                        j_p = (self.dic_col[ii, jj] - 1) % 61
                        if i_p == j_p:
                            if i_b != j_b and i_b == i_p:
                                cb1 = self.state_to_codon[j_b]
                                ca1 = self.state_to_codon[i_b]
                                if self.isNonsynonymous(cb1, ca1, self.codon_table):
                                    self.Q[ii, jj] = self.Q[ii, jj] - (self.tau * self.omega) + (tau * self.omega)
                                else:
                                    self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau
                            elif (i_b != j_b and j_b == j_p):
                                cb1 = self.state_to_codon[j_b]
                                ca1 = self.state_to_codon[i_b]
                                if self.isNonsynonymous(cb1, ca1, self.codon_table):
                                    self.Q[ii, jj] = self.Q[ii, jj] - (self.tau * self.omega) + (tau * self.omega)
                                else:
                                    self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau

            self.tau = tau

    def topo(self):

        ini=self.make_ini()
        list = []
        end1=deepcopy(ini)
        name_list=[]


        out_index=np.where(self.tree['process'] != scipy.stats.mode(self.tree['process'])[0])[0]

        branch_root_to_outgroup=self.tree['col'][out_index[0]]


        #print(len(self.tree['row']))

        length_edge=len(self.tree['row'])
        hash_node={}
        for i in range(length_edge+1):
            hash_node[i] = None

        hash_node[0]=deepcopy(ini)

        for i in range(length_edge):

            if i!=out_index:
                ini_index=self.tree['row'][i]
                end_index = self.tree['col'][i]


                if hash_node[end_index] is None:

                    print("ini node:", self.num_to_node[ini_index])
                    print("end node:", self.num_to_node[end_index])
                    ini_seq=deepcopy(hash_node[ini_index])
                    if self.tract_len is None:
                       if i==0:
                           end_seq = deepcopy(self.GLS_sequnce(ini=ini_seq,t=self.t[i], k=self.K, tau=self.fix_tau,iffirst=True))
                       else:
                           end_seq = deepcopy(
                               self.GLS_sequnce(ini=ini_seq, t=self.t[i], k=self.K, tau=self.fix_tau))


                     #  print(self.listprop.keys())
                     #  print(self.listprop.values())
                     #  print(self.listproptimes.keys())
                     #  print(self.listproptimes.values())

                    else:
                       end_seq = deepcopy(self.GLS_sequnce_tract(ini=ini_seq, t=self.t[i], k=self.K, tau=self.fix_tau))
                     #  print(self.listprop.keys())
                     #  print(self.listprop.values())
                      # print(self.listproptimes.keys())
                     #  print(self.listproptimes.values())

                    print("*******************************")

                    hash_node[end_index]=deepcopy(end_seq)
                    if end_index in set(self.observable_nodes):
                        list.append(end_seq)
                        name_list.append(end_index)

            else:
                # out group
                if self.Model == "HKY":
                    Q = self.remake_matrix()
                    end1 = np.ones(self.sizen)
                    Q_iiii = np.ones((4))
                    for ii in range(4):
                        qii = sum(Q[ii,])
                        if qii != 0:
                            Q_iiii[ii] = sum(Q[ii,])

                    for d in range(4):
                        Q[d,] = Q[d,] / Q_iiii[d]

                    for ll in range(self.sizen):
                        current_state = ini[ll] // 4
                        u = random.exponential(1 / Q_iiii[int(current_state)])
                        while (u <= self.t[i]):
                            a = np.random.choice(range(4), 1, p=Q[int(current_state),])[0]
                            current_state = a
                            u = u + random.exponential(1 / Q_iiii[int(current_state)])

                        end1[ll] = current_state




                else:
                    Q = self.remake_matrix()
                    Q_iiii = np.ones((61))
                    for ii in range(61):
                        qii = sum(Q[ii,])
                        if qii != 0:
                            Q_iiii[ii] = sum(Q[ii,])

                    for d in range(61):
                        Q[d,] = Q[d,] / Q_iiii[d]

                    for ll in range(self.sizen):
                        current_state = ini[ll] // 61
                        ifcontinue = True
                        u = 0
                        while (ifcontinue == True):
                            u = u + random.exponential(1 / Q_iiii[int(current_state)])
                            if (u <= self.t[i]):
                                a = np.random.choice(range(61), 1, p=Q[int(current_state),])[0]
                                current_state = a
                            else:
                                ifcontinue = False

                        end1[ll] = current_state

                end_index = self.tree['col'][i]

                hash_node[end_index] = deepcopy(end1)
        list.append(end1)
        name_list.append(branch_root_to_outgroup)



        return list,name_list


# translate the sequence with number level into site level
# 1 -> ATT

    def trans_into_seq(self,ini=None,name_list=None,casenumber=1):
        list = []



        if self.ifmakeQ==True and self.t is None:

            depth=self.leafnode

        else:
            depth = len(set(self.observable_nodes))-1

        if self.Model == 'MG94':
            dict = self.state_to_codon
            for i in range(depth):
                if i==0:
                   p0 = ">"+ self.num_to_node[name_list[i]] +"paralog0"+"\n"
                else:
                   p0 = "\n" + ">" + self.num_to_node[name_list[i]] + "paralog0" + "\n"

                p1 = "\n"+">"+ self.num_to_node[name_list[i]] +"paralog1"+"\n"
                for j in range(self.sizen):
                    p0 = p0 + dict[(ini[i][j]) // 61]
                    p1 = p1 + dict[(ini[i][j]) % 61]
                list.append(p0)
                list.append(p1)

            p0 = "\n"+">"+ self.num_to_node[name_list[(i+1)]] +"paralog0"+"\n"
            for j in range(self.sizen):
                p0 = p0 + dict[(ini[i+1][j])]
            list.append(p0)

        else:
            dict = self.state_to_nt
            for i in range(depth):
                if i==0:
                   p0 = ">"+self.num_to_node[name_list[i]]+"paralog0"+"\n"
                else:
                   p0 = "\n" + ">" + self.num_to_node[name_list[i]] + "paralog0" + "\n"
                p1 = "\n"+">"+self.num_to_node[name_list[i]]+"paralog1"+"\n"
                for j in range(self.sizen):
                    p0 = p0 + dict[(ini[i][j]) // 4]
                    p1 = p1 + dict[(ini[i][j]) % 4]
                list.append(p0)
                list.append(p1)


            p0 = "\n"+">"+self.num_to_node[name_list[(i+1)]]+"paralog0"+"\n"

            for j in range(self.sizen):
                p0 = p0 + dict[(ini[i+1][j])]
            list.append(p0)


        save_nameP = self.save_path + self.save_name+'.fasta'
        with open(save_nameP, 'wb') as f:
            for file in list:
               f.write(file.encode('utf-8'))

        for i in range(50):
            if self.hash_event_t[i] >0:
                print(i)
                print(self.hash_event[i]/(self.hash_event_t[i]*self.sizen*2))



        return (list)


    def measure_difference(self,ini,end):


        mutation_rate=0
        ini_paralog_div=0
        end_paralog_div = 0

        if self.Model=="MG94":
           str = {0}
           for i in range(61):
              d=i*61+i
              str.add(d)
        elif self.Model=="HKY":
            str = {0, 5, 10, 15}


        for i  in range(self.sizen):
            if(ini[i]!=end[i]):
                mutation_rate=mutation_rate+1
            if(ini[i] in str):
                ini_paralog_div=ini_paralog_div+1
            if(end[i]in str):
                end_paralog_div=end_paralog_div+1

        print("% tau at ini", self.fix_tau*np.power((ini_paralog_div/self.sizen),self.K))
        print("% tau at end", self.fix_tau*np.power((end_paralog_div/self.sizen),self.K))
        print("% identity between  paralogs at initial branch:", ini_paralog_div/self.sizen)
        print("% identity between  paralogs at ending branch:", end_paralog_div / self.sizen)
        print("% site difference between initial and ending branch per site:", mutation_rate/self.sizen)



    def compute_ave(self):
        len_dic=len(self.listproptimes.keys())
        out1=0
        out2=0
  #      for i in range(len_dic-1):
       #     out1=out1+(list(self.listprop.keys())[i]-list(self.listprop.keys())[i+1])*\
        #         (self.listprop[list(self.listprop.keys())[i+1]]/self.listproptimes[list(self.listprop.keys())[i+1]]+\
        #         self.listprop[list(self.listprop.keys())[i]]/self.listproptimes[list(self.listprop.keys())[i]])/2
        #     out2 = out2 + (list(self.listprop.keys())[i] - list(self.listprop.keys())[i + 1]) * \
        #           (1-(self.listprop[list(self.listprop.keys())[i + 1]] / self.listproptimes[
        #              list(self.listprop.keys())[i + 1]] + \
        #           self.listprop[list(self.listprop.keys())[i]] / self.listproptimes[
        #                list(self.listprop.keys())[i]]) / 2)

        out3=np.sum(list(self.listprop.values())[1:])/np.sum(list(self.listproptimes.values())[1:])


        return out3



    def proportion_change_IGC(self,repeats=3):
        pro_IGC=0
        for i in range(repeats):

            self.topo()
            out=self.compute_ave()
            pro_IGC =pro_IGC+out
            self.listprop={1:0}
            self.listproptimes = {1:0}


        print(pro_IGC/repeats)








if __name__ == '__main__':


        name = "YBL087C_YER117W_input"
        paralog = ['YBL087C', 'YER117W']
        alignment_file = '../test/yeast/' + name + '.fasta'
        newicktree = '../test/yeast/YeastTree.newick'
        save_path='../test/save/'

        Force = None
        model = 'MG94'

        type = 'situation_new'
        save_name = model + name
        geneconv = Embrachtau1(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                                  save_path=save_path, save_name=save_name)


        save_name_simu = model + name + "_simu"
        len_seq=geneconv.nsites


        self = GSseq(geneconv=geneconv, sizen=len_seq, ifmakeQ=False, Model=model, save_path=save_path,
                     save_name=save_name_simu, ifDNA=True)

        self.proportion_change_IGC()



        #    self = GSseq(geneconv,pi=[0.25,0.25,0.25,0.25],K=1.01,fix_tau=3.5,sizen=300,omega=1,leafnode=5,ifmakeQ=True)
     #   branch_list=[0.01,0.22,0.02,0.04,0.06,0.08,0.1,0.12,0.13,0.14,0.15,0.16]

   #     hashid={}
    #    for i in range(1):
    #        hashid[i] = 0

    #    for i in range(10):

 #       self = GSseq(newicktree=newicktree,sizen=3000,ifmakeQ=True,K=5,fix_tau=6,pi=[0.25,0.25,0.25,0.25],
     #                  kappa=1,Model=model,omega=1,save_path=save_path, save_name=save_name_simu,tract_len=10)


     #       self = GSseq(geneconv=geneconv, sizen=400, ifmakeQ=False,Model=model,save_path=save_path, save_name=save_name_simu)

       #     aaa=self.topo()
         #   self.trans_into_seq(ini=aaa[0],name_list=aaa[1])

   ##           if self.hash_event_t[j] > 0:
       ##  for i in range(50):
       #         print(i)
       #         print(hashid[i]/10)



    ##  paralog_simu = ['paralog0', 'paralog1']
     #   save_path1 = "./"
     #   save_name1=save_name_simu+"t2"


     #   geneconv_simu = Embrachtau1(newicktree, simulate_file, paralog_simu, Model=model, Force=Force, clock=None,
   #                            save_path=save_path1, save_name=save_name1)

    #    geneconv_simu.sum_branch(MAX=5,K=1.5)






