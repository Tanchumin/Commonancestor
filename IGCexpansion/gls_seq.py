# coding=utf-8
# A separate file for G_S to simulation by sequence level

# # Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function
import jsonctmctree.ll, jsonctmctree.interface
from CodonGeneconv import *
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from numpy import random
from scipy import linalg
import copy
from scipy.stats import poisson

from IGCexpansion.CodonGeneconFunc import isNonsynonymous
import pickle
import json
import numpy.core.multiarray


def get_maxpro(list, nodecom):
    sites = np.zeros(shape=(len(nodecom), len(list)))
    for site in range(len(list)):
        i=0
        for node in nodecom:
            sites[i][site] = np.argmax(list[site][:, node])
            i=i+1
    return (sites)


class GSseq:

    def __init__(self,
                 geneconv ,
                 sizen=111,branch_list=None,K=0.1,fix_tau=2
                 ):

        self.geneconv                 = geneconv
        self.ancestral_state_response = None
        self.scene                    = None
        self.num_to_state             = None
        self.num_to_node              = None
        self.node_length              = None
        self.dic_col                  = None

        self.codon_table = geneconv.codon_table
        self.tau =None
        self.omega=geneconv.omega
        self.Q_new = None
        self.Q= None

        self.sites1 = None
        self.sites2 = None
        self.sites=None
        self.dic_di = None
        self.tauoriginal=0

        self.node_length=0
        self.sites_length = self.geneconv.nsites
        self.Model=self.geneconv.Model
        self.ifmarginal = False

        self.min_diff=0
        self.igc_com=None

        self.judge=None
        self.P_list= None
        self.Q_original=None

        self.sizen=sizen
        self.t= branch_list
        self.fix_t=0.05
        self.K=K
        self.fix_tau=fix_tau





    def get_scene(self):
        self.tau=self.geneconv.tau
        if self.scene is None:
            self.geneconv.get_mle()
            self.scene = self.geneconv.get_scene()
        return self.scene

    def make_ini(self):
            ini = np.ones(self.sizen)
            z = self.geneconv.pi


            if self.Model=="HKY":
                sample=np.ones(4)
                for i in range(16):
                    if(i // 4 == i%4):
                        sample[i%4]=i

                for i in range(self.sizen):
                    ini[i] = int(np.random.choice(sample, 1, p=(z))[0])

            else:
                sample = np.ones(61)
                for i in range(3721):
                    if (i // 61 == i % 61):
                        sample[i % 61] = i

                for i in range(self.sizen):
                    ini[i] = int(np.random.choice(sample, 1, p=(1 / float(61)) * np.ones(61))[0])

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
            self.Q_new = np.zeros(shape=(16, 9))

        else:
            di = 3721
            di1 = 30
            self.Q_new = np.zeros(shape=(3721, 30))

        Q_iiii = np.ones((di))
        for ii in range(di):
            Q_iiii[ii] = sum(self.Q[ii,])

        for d in range(di):
            self.Q_new[d,] = self.Q[d,] / Q_iiii[d]

        return self.Q_new

    def making_Qmatrix(self):

        if self.scene is None:
            self.get_scene()

        scene = self.scene

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
            self.Q = np.zeros(shape=(3721, 30))
            self.dic_col = np.zeros(shape=(3721, 30))
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


    def GLS_sequnce(self, t=0.2, ini=None,k=1.1, tau=1):

        global di
        global di1

        if self.Model == "HKY":
            di = 16
            di1 = 9

        else:
            di = 3721
            di1 = 30

        u = 0

        while (u <= t):

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
            # print(p)
            change_location = np.random.choice(range(self.sizen), 1, p=(p/lambda_change))[0]
            change_site = int(ini[change_location])

            a = np.random.choice(range(di1), 1, p=self.Q_new[change_site,])[0]
            current_state = self.dic_col[change_site, a] - 1
            ini[change_location] = int(current_state)

        return ini

    def remake_matrix(self):
            if self.Model == "HKY":
                Q = self.geneconv.get_HKYBasic()
             #   print(Q)

            if self.Model == "MG94":
                Q = self.geneconv.get_MG94Basic()

            return Q

        # used  before topo so  that can make new Q
    def change_t_Q(self, tau=99):

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
                    for jj in range(30):
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
                                    self.Q[ii, jj] = self.Q[ii, jj] - (self.tau) + (tau)
                            elif (i_b != j_b and j_b == j_p):
                                cb1 = self.geneconv.state_to_codon[j_b]
                                ca1 = self.geneconv.state_to_codon[i_b]
                                if self.isNonsynonymous(cb1, ca1, self.geneconv.codon_table):
                                    self.Q[ii, jj] = self.Q[ii, jj] - (self.tau*self.omega) + (tau*self.omega)
                                else:
                                    self.Q[ii, jj] = self.Q[ii, jj] - (self.tau) + (tau)


            self.tau = tau

            # used  before topo so  that can make new Q

#   we derive sample tree
    def topo(self,leafnode=4):

        ini=self.make_ini()
        list = []

        if self.t is None:
            t=0.05
        else:
            t=self.t[1]/2
### build outgroup
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
                    curent_state = ini[ll]//4
                    u = random.exponential(1/Q_iiii[int(curent_state)])
                    while(u<=(2*t)):
                        a = np.random.choice(range(4), 1, p=Q[int(curent_state),])[0]
                        curent_state = a
                        u=u+random.exponential(1/Q_iiii[int(curent_state)])

                    end1[ll]=curent_state

            list1=[]
            list1.append(ini)
            mm=np.ones(shape=(4, self.sizen))
            mm[0,:]=ini

        elif self.Model=="MG94":
            Q = self.remake_matrix()
            end1 = np.ones(self.sizen)
            Q_iiii = np.ones((61))
            for ii in range(61):
                qii = sum(Q[ii,])
                if qii != 0:
                    Q_iiii[ii] = sum(Q[ii,])


            for d in range(61):
                Q[d,] = Q[d,] / Q_iiii[d]

            for ll in range(self.sizen):
                    curent_state = ini[ll]//61
                    u = random.exponential(1/Q_iiii[int(curent_state)])
                    while(u<=(2*t)):
                        a = np.random.choice(range(61), 1, p=Q[int(curent_state),])[0]
                        curent_state = a
                        u=u+random.exponential(1/Q_iiii[int(curent_state)])

                    end1[ll]=curent_state


            list1=[]
            list1.append(ini)
            mm=np.ones(shape=(4, self.sizen))
            mm[0,:]=ini

### start build internal node
        for i in range(leafnode):

            if(i== leafnode-1):
             #   print(ini)
                leaf = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t,k=self.K,tau=self.fix_tau))
                list.append(leaf)

            elif (i == leafnode - 2):
                # ini is internal node, leaf is observed;
                # list store observed
                ini = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t,k=self.K,tau=self.fix_tau))
                # self.change_t_Q(0.1)
                leaf = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t,k=self.K,tau=self.fix_tau))
                list.append(leaf)
                list1.append(ini)
                mm[i + 1, :] = ini

            else:
                # ini is internal node, leaf is observed;
                # list store observed
                ini = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t,k=self.K,tau=self.fix_tau))
            #    print(ini)
                leaf = deepcopy(self.GLS_sequnce(ini=deepcopy(ini),t=t,k=self.K,tau=self.fix_tau))
                list.append(leaf)
                list1.append(ini)
                mm[i + 1, :] = ini


        list.append(end1)


        return list


    def trans_into_seq(self,ini=None,leafnode=4):
        list = []
        if self.Model == 'MG94':
            dict = self.geneconv.state_to_codon
            for i in range(leafnode):
                p0 = ">paralog0"
                p1 = ">paralog1"
                for j in range(self.sizen):
                    p0 = p0 + dict[(ini[i][j]) // 61]
                    p1 = p1 + dict[(ini[i][j]) % 61]
                list.append(p0)
                list.append(p1)
        else:
            dict = self.geneconv.state_to_nt
            for i in range(leafnode):
                p0 = "\n"+">paralog0"+"\n"
                p1 = "\n"+">paralog1"+"\n"
                for j in range(self.sizen):
                    p0 = p0 + dict[(ini[i][j]) // 4]
                    p1 = p1 + dict[(ini[i][j]) % 4]
                list.append(p0)
                list.append(p1)

            p0 = "\n"+">paralog0"+"\n"
            for j in range(self.sizen):
                p0 = p0 + dict[(ini[leafnode][j])]

            list.append(p0)

        save_nameP = '../test/savesample/' + "FIX_k"+'sample1.txt'
        with open(save_nameP, 'wb') as f:
            pickle.dump(list, f)

        return (list)






if __name__ == '__main__':


        name = "YBL087C_YER117W_input"
        paralog = ['YBL087C', 'YER117W']
        alignment_file = '../test/yeast/' + name + '.fasta'
        newicktree = '../test/yeast/YeastTree.newick'

        Force = None
        model = 'MG94'

        type = 'situation_new'
        save_name = model + name
        geneconv = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                                   save_path='../test/save/', save_name=save_name)


        self = GSseq(geneconv)
        scene = self.get_scene()


        aaa=self.topo()
        print(self.trans_into_seq(ini=aaa))
    #    aaa=self.make_ini()
     #   print(self.GLS_sequnce(ini=aaa))







