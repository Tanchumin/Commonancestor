# coding=utf-8
# A separate file for G_S to simulation by sequence level

# # Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function
from em_pt1 import *
from copy import deepcopy
import os
import numpy as np
import scipy
import array
from numpy import random
from scipy import linalg
import copy
from scipy.stats import poisson

from IGCexpansion.CodonGeneconFunc import isNonsynonymous
import pickle
import numpy.core.multiarray


class GSseq:

    def __init__(self,
                 geneconv ,
                 sizen=400,branch_list=None,K=None,fix_tau=None,
                 pi=None,omega=None,kappa=None,leafnode=4,
                 ifmakeQ=False,
                 ):

        self.geneconv                 = geneconv
        self.ancestral_state_response = None
        self.scene                    = None
        self.num_to_state             = None
        self.num_to_node              = None
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
        self.sites_length = self.geneconv.nsites
        self.Model=self.geneconv.Model
        self.sites = None


#making Q matrix
        self.codon_nonstop = self.geneconv.codon_nonstop
        self.codon_to_state=self.geneconv.codon_to_state
        self.codon_table=self.geneconv.codon_table

## simulation setting
        self.sizen=sizen
        self.t= branch_list
        self.K=K
## self.tau is used to update for Q matriex
        self.tau =fix_tau
        self.fix_tau=fix_tau
        self.leafnode=leafnode



        self.initialize_parameters()



    def initialize_parameters(self):

        self.tree=deepcopy(self.geneconv.tree)


        if self.ifmakeQ==False:
            print("The parameters and tree are inherited from inputs")
            self.scene=self.geneconv.read_parameter_gls()
            self.pi = deepcopy(self.geneconv.pi)


        if self.fix_tau is None:
            self.fix_tau = self.geneconv.tau
            self.tau=self.geneconv.tau

        if self.K is None:
            self.K = self.geneconv.K

        if self.omega is None:
            self.omega = self.geneconv.omega

        if self.kappa is None:
            self.kappa=self.geneconv.kappa

        if self.pi is None:
            self.pi = self.geneconv.pi






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
        expected_rate = np.dot(self.geneconv.prior_distribution, Qbasic.sum(axis=1))
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
            sa = self.geneconv.nt_to_state[na]
            sb = self.geneconv.nt_to_state[nb]
            for j, pair_to in enumerate(product('ACGT', repeat=2)):
                nc, nd = pair_to
                sc = self.geneconv.nt_to_state[nc]
                sd = self.geneconv.nt_to_state[nd]
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


    def get_prior(self):

        if self.Model == 'MG94':
            self.prior_feasible_states = [(self.codon_to_state[codon], self.codon_to_state[codon]) for codon in
                                          self.codon_nonstop]
            distn = [reduce(mul, [self.pi['ACGT'.index(b)] for b in codon], 1) for codon in self.codon_nonstop]
            distn = np.array(distn) / sum(distn)
        elif self.Model == 'HKY':
            self.prior_feasible_states = [(self.geneconv.nt_to_state[nt], self.geneconv.nt_to_state[nt]) for nt in 'ACGT']
            distn = [self.pi['ACGT'.index(nt)] for nt in 'ACGT']
            distn = np.array(distn) / sum(distn)

        self.prior_distribution = distn

    def isTransition(self,na, nb):
       return (set([na, nb]) == set(['A', 'G']) or set([na, nb]) == set(['C', 'T']))




    def get_station_dis(self):
        pi=np.zeros(61)
        for ca in self.geneconv.codon_nonstop:
            p=1
            for i in range(3):
                p=p*self.geneconv.pi['ACGT'.index(ca[i])]
            pi[int(self.geneconv.codon_to_state[ca])]=p

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
            else:
                for i in range(self.sizen):
                    ca = (ini[i]) // 61
                    cb = (ini[i]) % 61
                    if ca != cb:
                        index = index + 1

            index = 1 - (index / self.sizen)

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
                igc = (self.tau) / qq
                point=1-igc

            elif (i_b != j_b and j_b == j_p):
                y_coor = np.argwhere(self.dic_col[int(pre),] == (int(post) + 1))[0]
                qq = self.Q[int(pre), y_coor]
                igc = (self.tau) / qq
                point = 1 - igc
        else:
            point=1

        if i_p!=i_b:
            change_i=1
        else:
            change_j = 1



        return point,igc,change_i,change_j


# GLS algorithm on sequence level, which just generate on ini and end sequence

    def GLS_sequnce(self, t=0.1, ini=None,k=1.1, tau=1.1):

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

            for d in range(di):
                self.Q_new[d,] = self.Q[d,] / Q_iiii[d]

            p = np.zeros(self.sizen)
            lambda_change = 0
            for ll in range(self.sizen):
                lambda_change = Q_iiii[int(ini[ll])] + lambda_change
                p[ll] = Q_iiii[int(ini[ll])]


            u = u + random.exponential(1/lambda_change)
            if (u <= t):

                change_location = np.random.choice(range(self.sizen), 1, p=(p/lambda_change))[0]
                change_site = int(ini[change_location])



                a = np.random.choice(range(di1), 1, p=self.Q_new[change_site,])[0]
                current_state = self.dic_col[change_site, a] - 1
                result=self.point_IGC(change_site,current_state)
                igc=result[1]+igc
                point=result[0]+point
                change_j=result[3]+change_j
                change_i = result[2] + change_i
                ini[change_location] = int(current_state)

            else:
                ifcontinue=False




        print("% branch length", t)
        print("% estimated number of point mutation number per site:", point / self.sizen)
        print("% estimated number of IGC number per site:", igc / self.sizen)
        print("% estimated number of one paralog  change per site:", change_i / self.sizen)
        print("% estimated number of the other paralog  change per site:", change_j / self.sizen)
        self.measure_difference(ini=deepcopy(inirel),end=deepcopy(ini))

        return ini

    def remake_matrix(self):
            if self.Model == "HKY":
                Q = self.get_HKYBasic()
             #   print(Q)

            if self.Model == "MG94":
                Q = self.get_MG94Basic()

            return Q

# use change_t_Q before topo so  that can make new Q conditioned on (tau^id)*K at each GLS step
    def change_t_Q(self, tau=0.99):

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
                                cb1 = self.geneconv.state_to_codon[j_b]
                                ca1 = self.geneconv.state_to_codon[i_b]
                                if self.isNonsynonymous(cb1, ca1, self.geneconv.codon_table):
                                    self.Q[ii, jj] = self.Q[ii, jj] - (self.tau*self.omega) + (tau*self.omega)
                                else:
                                    self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau
                            elif (i_b != j_b and j_b == j_p):
                                cb1 = self.geneconv.state_to_codon[j_b]
                                ca1 = self.geneconv.state_to_codon[i_b]
                                if self.isNonsynonymous(cb1, ca1, self.geneconv.codon_table):
                                    self.Q[ii, jj] = self.Q[ii, jj] - (self.tau*self.omega) + (tau*self.omega)
                                else:
                                    self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau

            self.tau = tau



    def topo(self):

        ini=self.make_ini()
        list = []
        end1=deepcopy(ini)
        list1 = []
        name_list=[]


        if self.ifmakeQ==False:
                t=self.geneconv.tree['rate']

        else:
            if self.t is None:

                t = 0.04
                print("Model assumes all branch with the same branch length", t)
            else:
                t=self.t


        if self.ifmakeQ==True and self.t is None:

            if self.Model=="HKY":
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
                        current_state = ini[ll]//4
                        u = random.exponential(1/Q_iiii[int(current_state)])
                        while(u<=(2*t)):
                            a = np.random.choice(range(4), 1, p=Q[int(current_state),])[0]
                            current_state = a
                            u=u+random.exponential(1/Q_iiii[int(current_state)])

                        end1[ll]=current_state

                list1=[]
                list1.append(ini)


            elif self.Model=="MG94":
                Q = self.remake_matrix()
                Q_iiii = np.ones((61))
                for ii in range(61):
                    qii = sum(Q[ii,])
                    if qii != 0:
                        Q_iiii[ii] = sum(Q[ii,])

                for d in range(61):
                    Q[d,] = Q[d,] / Q_iiii[d]

                for ll in range(self.sizen):
                        current_state = ini[ll]//61
                        ifcontinue = True
                        u=0
                        while (ifcontinue == True):
                            u = u+random.exponential(1 / Q_iiii[int(current_state)])
                            if (u<=t):
                                a = np.random.choice(range(61), 1, p=Q[int(current_state),])[0]
                                current_state = a
                            else:
                                ifcontinue=False

                        end1[ll]=current_state


                list1.append(ini)



    ### start build internal node
            for i in range(self.leafnode):

                if(i== self.leafnode-1):
                 #   print(ini)
                    leaf = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t,k=self.K,tau=self.fix_tau))
                    list.append(leaf)

                elif (i == self.leafnode - 2):
                    # ini is internal node, leaf is observed;
                    # list store observed
                    ini = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t,k=self.K,tau=self.fix_tau))
                    # self.change_t_Q(0.1)
                    leaf = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t,k=self.K,tau=self.fix_tau))
                    list.append(leaf)
                    list1.append(ini)

                else:
                    # ini is internal node, leaf is observed;
                    # list store observed
                    if (i==0):
                        ini = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t/3,k=self.K,tau=self.fix_tau))
                    else:
                        ini = deepcopy(self.GLS_sequnce(ini=deepcopy(ini), t=t, k=self.K, tau=self.fix_tau))
                #    print(ini)
                    leaf = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t,k=self.K,tau=self.fix_tau))
                    list.append(leaf)
                    list1.append(ini)

            list.append(end1)

        else:

            out_index=np.where(self.tree['process'] != scipy.stats.mode(self.tree['process'])[0])[0]

            branch_root_to_outgroup=self.geneconv.tree['col'][out_index[0]]




            length_edge=len(self.tree['row'])
            hash_node={}
            for i in range(length_edge+1):
                hash_node[i] = None

            hash_node[0]=deepcopy(ini)


            for i in range(length_edge):

                if i!=out_index:
                    ini_index=self.geneconv.tree['row'][i]
                    end_index = self.geneconv.tree['col'][i]


                    if hash_node[end_index] is None:
                        print("ini node is", self.geneconv.num_to_node[ini_index])
                        print("end node is", self.geneconv.num_to_node[end_index])
                        ini_seq=deepcopy(hash_node[ini_index])
                        end_seq = deepcopy(self.GLS_sequnce(ini=ini_seq,t=t[i], k=self.K, tau=self.fix_tau))

                        print("*******************************")

                        hash_node[end_index]=deepcopy(end_seq)
                        if end_index in set(self.geneconv.observable_nodes):
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
                            while (u <= (2 * t[i])):
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
                                if (u <= t[i]):
                                    a = np.random.choice(range(61), 1, p=Q[int(current_state),])[0]
                                    current_state = a
                                else:
                                    ifcontinue = False

                            end1[ll] = current_state

                    end_index = self.geneconv.tree['col'][i]

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
            depth = len(set(geneconv.observable_nodes))-1

        if self.Model == 'MG94':
            dict = self.geneconv.state_to_codon
            for i in range(depth):
                if i==0:
                   p0 = ">"+ self.geneconv.num_to_node[name_list[i]] +"paralog0"+"\n"
                else:
                   p0 = "\n" + ">" + self.geneconv.num_to_node[name_list[i]] + "paralog0" + "\n"

                p1 = "\n"+">"+ self.geneconv.num_to_node[name_list[i]] +"paralog1"+"\n"
                for j in range(self.sizen):
                    p0 = p0 + dict[(ini[i][j]) // 61]
                    p1 = p1 + dict[(ini[i][j]) % 61]
                list.append(p0)
                list.append(p1)

            p0 = "\n"+">"+ self.geneconv.num_to_node[name_list[(i+1)]] +"paralog0"+"\n"
            for j in range(self.sizen):
                p0 = p0 + dict[(ini[i][j])]

            list.append(p0)

        else:
            dict = self.geneconv.state_to_nt
            for i in range(depth):
                if i==0:
                   p0 = ">"+self.geneconv.num_to_node[name_list[i]]+"paralog0"+"\n"
                else:
                   p0 = "\n" + ">" + self.geneconv.num_to_node[name_list[i]] + "paralog0" + "\n"
                p1 = "\n"+">"+self.geneconv.num_to_node[name_list[i]]+"paralog1"+"\n"
                for j in range(self.sizen):
                    p0 = p0 + dict[(ini[i][j]) // 4]
                    p1 = p1 + dict[(ini[i][j]) % 4]
                list.append(p0)
                list.append(p1)


            p0 = "\n"+">"+self.geneconv.num_to_node[name_list[(i+1)]]+"paralog0"+"\n"

            for j in range(self.sizen):

                p0 = p0 + dict[(ini[i+1][j])]

            list.append(p0)


        save_nameP = '../test/savesample/' + "FIX_k_"+str(casenumber)+'_sample.fasta'
        with open(save_nameP, 'wb') as f:
            for file in list:
               f.write(file.encode('utf-8'))



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
        print("% site difference between initial and ending branch per site", mutation_rate/self.sizen)







if __name__ == '__main__':


        name = "YBL087C_YER117W_input"
        paralog = ['YBL087C', 'YER117W']
        alignment_file = '../test/yeast/' + name + '.fasta'
        newicktree = '../test/yeast/YeastTree.newick'

        Force = None
        model = 'HKY'

        type = 'situation_new'
        save_name = model + name
        geneconv = Embrachtau1(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                                   save_path='../test/save/', save_name=save_name,if_rerun=False)


    #    self = GSseq(geneconv,pi=[0.25,0.25,0.25,0.25],K=1.01,fix_tau=3.5,sizen=300,omega=1,leafnode=5,ifmakeQ=True)
        self = GSseq(geneconv, ifmakeQ=False)

        aaa=self.topo()
        self.trans_into_seq(ini=aaa[0],name_list=aaa[1])
   #     print(set(self.geneconv.observable_nodes))

     #   print(self.GLS_sequnce(ini=aaa))


#[0.98735, 0, 0.9398, 0.941275, 0.8757 ,0.875625, 0.81945, 0.820125,  0.770575 ,0.771875]




