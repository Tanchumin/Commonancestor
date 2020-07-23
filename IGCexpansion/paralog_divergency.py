 #A separate file for Ancestral State Reconstruction
# Uses Alex Griffing's JsonCTMCTree package for likelihood and gradient calculation
# Xiang Ji
# xji3@ncsu.edu
# Tanchumin Xu
# txu7@ncsu.edu
from __future__ import print_function
import jsonctmctree.ll, jsonctmctree.interface
from CodonGeneconv import *
from copy import deepcopy
import os
from Common import *
import numpy as np
import pandas as pd
from numpy import random


def get_maxpro(list, nodecom):
    sites = np.zeros(shape=(len(nodecom), len(list)))
    for site in range(len(list)):
        i=0
        for node in nodecom:
            sites[i][site] = np.argmax(list[site][:, node])
            i=i+1
    return (sites)


class AncestralState:

    def __init__(self,
                 geneconv  # JSGeneconv analysis for now
                 ):

        self.geneconv                 = geneconv
        self.ancestral_state_response = None
        self.scene                    = None
        self.num_to_state             = None
        self.num_to_node              = None
        self.node_length              = None
        self.dic_col                  = None

        self.codon_table = geneconv.codon_table
        self.tau =geneconv.tau1
        self.omega=geneconv.omega
        self.Q_new = None
        self.Q= None

        self.sites1 = None
        self.sites2 = None
        self.dic_di = None

        # history from monte carol
        self.time_list= None
        self.state_list = None
        self.effect_matrix= None
        self.big_number_matrix= None
        self.dis_matrix = None

        self.node_length=0
        self.sites_length = self.geneconv.nsites
        self.Model=self.geneconv.Model
        self.ifmarginal = False


    def get_mle(self):
        self.geneconv.get_mle()

    def get_scene(self):
        if self.scene is None:
            self.get_mle()
            self.scene = self.geneconv.get_scene()
        return self.scene

    def get_dict_trans(self):
        return self.geneconv.get_dict_trans()

    def get_ancestral_state_response(self):
        scene = self.get_scene()
        requests = [
            {'property': "DNDNODE"}
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
            j_out = jsonctmctree.interface.process_json_in(j_in)
            if j_out['status'] is 'feasible':
                result = j_out['responses'][0]
            else:
                raise RuntimeError('Failed at obtaining ancestral state distributions.')
        return result

    def get_joint_matrix(self,node):
        if self.ancestral_state_response is None:
            self.ancestral_state_response = self.get_ancestral_state_response()

        if self.Model=='MG94':
            sites = np.zeros(shape=(3721, self.sites_length))
            for site in range(self.sites_length):
                for pr in range(3721):
                    sites[pr][site] = self.ancestral_state_response[site][pr, node]
        else:
            sites = np.zeros(shape=(16, self.sites_length))
            for site in range(self.sites_length):
                for pr in range(16):
                    sites[pr][site] = self.ancestral_state_response[site][pr][node]

        return (sites)

    def get_site_ancestral_dist(self, site_num):
        # site_num starts from 0 where 0 corresponds to the first site
        if self.ancestral_state_response is None:
            self.ancestral_state_response = self.get_ancestral_state_response()

        site_packed_dist = self.ancestral_state_response[site_num]
        node_state_prob_dict = {}
        for node_num in range(len(self.get_num_to_node())):
            state_prob_dict = {}
            for state_num in range(len(self.get_num_to_state())):
                node_state_prob = site_packed_dist[state_num][node_num]
                state_prob_dict[self.num_to_state[state_num]] = node_state_prob
            node_state_prob_dict[self.num_to_node[node_num]] =state_prob_dict
        return node_state_prob_dict

    def get_maxpro_index(self, ifmarginal=False, paralog=1):

        if self.ancestral_state_response is None:
            self.ancestral_state_response = self.get_ancestral_state_response()

        if ifmarginal == False:
            self.node_length=len(self.get_num_to_node())
            sites = np.zeros(shape=(self.node_length,self.sites_length ))
            for site in range(self.sites_length):
                for node in range(self.node_length):
                     sites[node][site] = np.argmax(np.array(self.ancestral_state_response[site])[:, node])

        else:
            if paralog==1:
                self.node_length = len(self.get_num_to_node())
                sites = np.zeros(shape=(self.node_length, self.sites_length))
                for node in range(self.node_length):
                    mat=self.get_marginal(node)
                    for site in range(self.sites_length):
                        sites[node][site] = np.argmax(np.array(mat)[site,:])
                self.sites1 = sites

            else:
                self.node_length = len(self.get_num_to_node())
                sites = np.zeros(shape=(self.node_length, self.sites_length))
                for node in range(self.node_length):
                    mat=self.get_marginal(node,paralog)
                    for site in range(self.sites_length):
                        sites[node][site] = np.argmax(np.array(mat)[site,:])
                self.sites2 = sites

        return (sites)

    def get_maxpro_matrix(self, ifmarginal=False, paralog=1):

        if self.ancestral_state_response is None:
            self.ancestral_state_response = self.get_ancestral_state_response()

        if ifmarginal == False:
            self.node_length=len(self.get_num_to_node())
            sites = np.zeros(shape=(self.node_length, self.sites_length))
            for site in range(self.sites_length):
                for node in range(self.node_length):
                    sites[node][site] = np.max(np.array(self.ancestral_state_response[site])[:, node])

        else:
            if paralog == 1:
                self.node_length = len(self.get_num_to_node())
                sites = np.zeros(shape=(self.node_length, self.sites_length))
                for node in range(self.node_length):
                    mat=self.get_marginal(node)
                    for site in range(self.sites_length):
                        sites[node][site] = np.max(np.array(mat)[site,:])

            else:
                self.node_length = len(self.get_num_to_node())
                sites = np.zeros(shape=(self.node_length, self.sites_length))
                for node in range(self.node_length):
                    mat=self.get_marginal(node,2)
                    for site in range(self.sites_length):
                        sites[node][site] = np.max(np.array(mat)[site,:])



        return (sites)

    def translate_into_seq(self,ifmarginal=False,paralog=1):
        promax=self.get_maxpro_index(ifmarginal,paralog)
        list = []

        if self.Model == 'MG94':
            dict = self.geneconv.state_to_codon
            for i in range(self.node_length):
                p0 = "paralog0:"
                p1 = "paralog1:"
                for j in range(self.sites_length):
                    p0 = p0 + dict[(promax[i][j]) // 61]
                    p1 = p1 + dict[(promax[i][j]) % 61]
                list.append(p0)
                list.append(p1)
        else:
            dict = self.geneconv.state_to_nt
            for i in range(self.node_length):
                p0 = "paralog0:"
                p1 = "paralog1:"
                for j in range(self.sites_length):
                    p0 = p0 + dict[(promax[i][j]) // 4]
                    p1 = p1 + dict[(promax[i][j]) % 4]
                list.append(p0)
                list.append(p1)

        return (list)

    def get_num(self):
        if self.num_to_state is None:
            if self.Model == 'HKY':
                states = 'ACGT'
            elif self.Model == 'MG94':
                states = geneconv.codon_to_state

        return(states)


    def get_num_to_state(self):
        if self.num_to_state is None:
            if self.Model == 'HKY':
                states = 'ACGT'
            elif self.Model == 'MG94':
                states = geneconv.codon_nonstop
            self.num_to_state = {num:state for num, state in enumerate(product(states, repeat = 2))}
            self.num_to_state = {num:state for num, state in enumerate(product(states, repeat = 2))}
        return self.num_to_state

    def get_num_to_node(self):
        if self.num_to_node is None:
            self.num_to_node = self.geneconv.num_to_node
        return self.num_to_node

    def get_marginal(self, node, paralog=1):
        if self.ancestral_state_response is None:
            self.ancestral_state_response = self.get_ancestral_state_response()

        if paralog == 1:

            if self.Model == 'MG94':
                marginal_sites = np.zeros(shape=(self.sites_length,61))
                for site in range(self.sites_length):
                    i = 0
                    for marginal in range(61):
                        marginal_sites[site][marginal] = sum(np.array(self.ancestral_state_response[site])[
                                                             i:(i+61), node])
                        i=i+61

            else:
                marginal_sites = np.zeros(shape=(self.sites_length, 4))
                for site in range(self.sites_length):
                    i = 0
                    for marginal in range(4):
                        marginal_sites[site][marginal] = sum(np.array(self.ancestral_state_response[site])[
                                                             i:(i + 4), node])
                        i = i + 4
        else:
            if self.Model == 'MG94':
                marginal_sites = np.zeros(shape=(self.sites_length, 61))
                for site in range(self.sites_length):
                    i = 0
                    for marginal in range(61):
                        index_pra2 = range(i, 3671+i, 61)
                        marginal_sites[site][marginal] = sum(np.array(self.ancestral_state_response[site])[
                                                                 index_pra2, node])
                        i = i + 1

            else:
                marginal_sites = np.zeros(shape=(self.sites_length, 4))
                for site in range(self.sites_length):
                    i = 0
                    for marginal in range(4):
                        index_pra2=range(i,i+16,4)
                        marginal_sites[site][marginal] = sum(np.array(self.ancestral_state_response[site])[
                                                                 index_pra2, node])
                        i = i + 1

        return marginal_sites

    def get_interior_node(self):

        if self.node_length is None:
             self.node_length = len(self.get_num_to_node())

        node = np.arange(self.node_length)
        interior_node = set(node) - set(self.scene["observed_data"]["nodes"])
        c = [i for i in interior_node]

        return (c)


# help build dictionary for how difference of a paralog

    def making_dic_di(self):

        if self.Model == "HKY":
            dicts = {}
            keys = range(16)
            for i in keys:
                if (i//4) == (i % 4):
                    dicts[i] = 0
                else:
                    dicts[i] = 1

        else:
            dicts = {}
            keys = range(3721)
            for i in keys:
                if (i//61) == (i % 61):
                    dicts[i] = 0
                else:
                    dicts[i] = 1

        self.dic_di = dicts

# making Q matrix
    def making_Qmatrix(self):

        scene = self.get_scene()
        actual_number=(len(scene['process_definitions'][1]['transition_rates']))

        global x_i
        x_i = 0

        global index
        index = 0

# dic is from 1:16

        if self.Model == 'HKY':
            self.Q=np.zeros(shape=(16, 9))
            self.dic_col=np.zeros(shape=(16, 9))
            for i in range(actual_number):

# x_io means current index for row states,x_i is states for last times
# self.dic_col indicts the coordinates for ending states

                x_io = (scene['process_definitions'][1]['row_states'][i][0])*4+(scene['process_definitions'][1][
                        'row_states'][i][1])

                if x_i == x_io:
                    self.Q[x_io, index] = scene['process_definitions'][1]['transition_rates'][i]
                    self.dic_col[x_io, index] = 1 + (scene['process_definitions'][1]['column_states'][i][0]) * 4+(
                                               scene['process_definitions'][1]['column_states'][i][1])
                    x_i = x_io
                    index = index+1

                else:
                    self.Q[x_io, 0]=scene['process_definitions'][1]['transition_rates'][i]
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
                    self.dic_col[x_io, 0] = 1+(scene['process_definitions'][1]['column_states'][i][0]) * 61 + (
                                            scene['process_definitions'][1]['column_states'][i][1])
                    x_i = x_io
                    index = 1


        return self.Q, self.dic_col


# making initial state and end state

    def make_ie(self, node_i, node_e):

        if self.sites1 is None:
            self.get_maxpro_index(True, 1)
        if self.sites2 is None:
            self.get_maxpro_index(True, 2)

        ini = np.ones(self.sites_length)
        end = np.ones(self.sites_length)

        if self.Model == "HKY":
            for site in range(self.sites_length):
               ini[site] = self.sites2[node_i, site]+self.sites1[node_i, site]*4
               end[site] = self.sites2[node_e, site]+self.sites1[node_e, site]*4

        else:
            for site in range(self.sites_length):
               ini[site] = self.sites2[node_i, site]+self.sites1[node_i, site]*61
               end[site] = self.sites2[node_e, site]+self.sites1[node_e, site]*61

        return ini, end



    def GLS( self,t=0.36):

        # Q_iiii is diagnal of Q
        # max is biggest change in a line
        global di
        global di1

        if self.Model == "HKY":

            di=16
            di1=9
            self.Q_new=np.zeros(shape=(16,9))
        else:
            di=3721
            di1=27
            self.Q_new = np.zeros(shape=(3721, 27))

        if self.Q is None:
           self.making_Qmatrix()

        Q_iiii = np.ones((di))
        for ii in range(di):
            qii=sum(self.Q[ii,])
            if qii !=0:
               Q_iiii[ii] = sum(self.Q[ii,])


        max_number = 10
        time_matrix = 100 * np.ones(shape=(self.sites_length, max_number))
        state_matrix = np.zeros(shape=(self.sites_length, max_number))



        for d in range(di):
            self.Q_new[d,] = self.Q[d,] / Q_iiii[d]

        # how many change in history
        # big_number how to trunct the matrix
        effect_number = 0
        big_number = 0

        ini=self.make_ie(1,2)[0]
        end=self.make_ie(1,2)[1]





        for ll in range(self.sites_length):
            # most transfer 5 times

            i = 0
            time = [100]
            state = [0]
            curent_state = ini[ll]



            while (curent_state != end[ll]):
                curent_state = ini[ll]
                u = 0
                i = 0
                time = [100]
                state = [0]
                while (u <= t):
                    # state may from 0 or 1
                    i = i + 1
                    u1 = random.exponential(Q_iiii[int(curent_state)])
                    u = u + u1
                    time.append(u)
                    a = np.random.choice(range(di1), 1, p=self.Q_new[int(curent_state),])[0]
                    old=curent_state
                    curent_state = self.dic_col[int(curent_state),a]-1
                    while (sum(self.Q_new[int(curent_state),]) == 0  and u <= t):
                        a = np.random.choice(range(di1), 1, p=self.Q_new[int(old),])[0]
                        curent_state = self.dic_col[int(old), a] - 1
                    state.append(int(curent_state))



            if (i > max_number):
                big_number = i
                time_matrix_old = time_matrix
                state_matrix_old = state_matrix
                time_matrix = np.zeros(shape=(self.sites_length, i))
                state_matrix = np.zeros(shape=(self.sites_length, i))
                time_matrix[0:self.sites_length, 0:max_number] = time_matrix_old
                state_matrix[0:self.sites_length, 0:max_number] = state_matrix_old
                time_matrix[ll, 0:i] = time[0:i]
                state_matrix[ll, 0:i] = state[0:i]

                max_number = big_number
            else:
                big_number = max(big_number, i)
                if (i > 0):
                    effect_number = (i - 1) + effect_number
                time_matrix[ll, 0:i] = time[0:i]
                state_matrix[ll, 0:i] = state[0:i]
                state_matrix[ll, 0] = ini[ll]

            print(ll)


        time_matrix = time_matrix[0:self.sites_length, 0:big_number]
        state_matrix = state_matrix[0:self.sites_length, 0:big_number]

        return time_matrix, state_matrix, effect_number, big_number

    def GLS_m(self,t=0.1, ini=None, end=None, repeat=2):

        global di
        global di1

        if self.Model == "HKY":

            di=16
            di1=9
            self.Q_new=np.zeros(shape=(16, 9))
        else:
            di=3721
            di1=27
            self.Q_new = np.zeros(shape=(3721, 27))

        if self.Q is None:
           self.making_Qmatrix()

        Q_iiii = np.ones(di)
        for ii in range(di):
            qii = sum(self.Q[ii, ])
            if qii !=0:
               Q_iiii[ii] = qii

        for d in range(di):
            self.Q_new[d, ] = self.Q[d, ] / Q_iiii[d]

        max_number = 10

        time_list = []
        state_list = []
        dis_matrix = np.ones(self.sites_length)

        for i in range(self.sites_length):
            if ini[i] != end[i]:
                time_list.append(0)
                state_list.append(0)
                dis_matrix[i] = 0
            else:
                time_list.append(1)
                state_list.append(1)
        effect_matrix = np.zeros(shape=(self.sites_length, repeat))
        big_number_matrix=np.zeros(shape=(self.sites_length, repeat))

        # start simulation

        for ii in range(self.sites_length):

            # time_list ! =1 means there is a mutation

            if time_list[ii] != 1:

                time_matrix = 100*np.ones(shape=(repeat, max_number))
                state_matrix = np.zeros(shape=(repeat, max_number))

                for jj in range(repeat):
                        # most transfer 10 times
                        current_state = ini[ii]
                        i = 0
                        time = [0]
                        state = [0]

                        effect_number = 0
                        big_number = 0

                        while current_state != end[ii]:
                            current_state = ini[ii]
                            i = 1
                            time = [100]
                            state = [0]
                            u1 = np.random.uniform(0,1)
                            u= -np.log(1-(1-np.exp(-Q_iiii[int(current_state)]*t))*u1)/\
                                       (Q_iiii[int(current_state)])
                            time.append(u)
                            a = np.random.choice(range(di1), 1, p=self.Q_new[int(current_state), ])[0]
                            old = current_state
                            current_state = self.dic_col[int(current_state), a] - 1

                                    # if jump to absorbing state and without finishing process, we need to resample

                            while sum(self.Q_new[int(current_state), ]) == 0:
                                    a = np.random.choice(range(di1), 1, p=self.Q_new[int(old), ])[0]
                                    current_state = self.dic_col[int(old), a] - 1
                            state.append(int(current_state))

                            while u<=t:
                                    i=i+1
                                    u1 = random.exponential(Q_iiii[int(current_state)])
                                    u = u + u1
                                    time.append(u)
                                    a = np.random.choice(range(di1), 1, p=self.Q_new[int(current_state), ])[0]
                                    old = current_state
                                    current_state = self.dic_col[int(current_state), a] - 1

                                    # if jump to absorbing state and without finishing process, we need to resample

                                    while sum(self.Q_new[int(current_state), ]) == 0:
                                        a = np.random.choice(range(di1), 1, p=self.Q_new[int(old), ])[0]
                                        current_state = self.dic_col[int(old), a] - 1
                                    state.append(int(current_state))
                            current_state = state[i - 1]


                        if i > max_number:
                            big_number = i
                            time_matrix_old = time_matrix
                            state_matrix_old = state_matrix
                            time_matrix = np.zeros(shape=(self.sites_length, i))
                            state_matrix = np.zeros(shape=(self.sites_length, i))
                            time_matrix[0:self.sites_length, 0:max_number] = time_matrix_old
                            state_matrix[0:self.sites_length, 0:max_number] = state_matrix_old
                            time_matrix[jj, 0:i] = time[0:i]
                            state_matrix[jj, 0:i] = state[0:i]
                        else:
                            big_number = max(big_number, i)
                            if i > 0:
                                effect_number = (i - 1) + effect_number
                            time_matrix[jj, 0:i] = time[0:i]
                            state_matrix[jj, 0:i] = state[0:i]
                            state_matrix[jj, 0] = ini[ii]
                        effect_matrix[ii,jj] = int(effect_number)
                        big_number_matrix[ii,jj] = int(big_number)


                time_list[ii]=time_matrix
                state_list[ii]=state_matrix

            else:

                time_matrix = 100 * np.ones(shape=(repeat, max_number))
                state_matrix = np.zeros(shape=(repeat, max_number))

                for jj in range(repeat):
                    # most transfer 10 times
                    current_state = ini[ii]
                    i = 0

                    effect_number = 0
                    big_number = 0

                    u = 0
                    time = [100]
                    state = [0]
                    while u<=t:
                        u1 = random.exponential(Q_iiii[int(current_state)])
                        u = u + u1
                        i=i+1
                        if u<=t:
                            time.append(u)
                            a = np.random.choice(range(di1), 1, p=self.Q_new[int(current_state),])[0]
                            old = current_state
                            current_state = self.dic_col[int(current_state), a] - 1

                                # if jump to absorbing state and without finishing process, we need to resample

                            while sum(self.Q_new[int(current_state),]) == 0:
                              a = np.random.choice(range(di1), 1, p=self.Q_new[int(old),])[0]
                              current_state = self.dic_col[int(old), a] - 1
                            state.append(int(current_state))
                    current_state = state[i - 1]
                    if current_state != end[ii]:
                        state = [0]
                        effect_number = 0
                        big_number = 0
                        time=[100]

                    if i > max_number:
                        big_number = i
                        time_matrix_old = time_matrix
                        state_matrix_old = state_matrix
                        time_matrix = np.zeros(shape=(self.sites_length, i))
                        state_matrix = np.zeros(shape=(self.sites_length, i))
                        time_matrix[0:self.sites_length, 0:max_number] = time_matrix_old
                        state_matrix[0:self.sites_length, 0:max_number] = state_matrix_old
                        time_matrix[jj, 0:i] = time[0:i]
                        state_matrix[jj, 0:i] = state[0:i]
                    else:
                        big_number = max(big_number, i)
                        if i > 0:
                            effect_number = (i - 1) + effect_number
                        time_matrix[jj, 0:i] = time[0:i]
                        state_matrix[jj, 0:i] = state[0:i]
                        state_matrix[jj, 0] = ini[ii]
                    effect_matrix[ii, jj] = int(effect_number)
                    big_number_matrix[ii, jj] = int(big_number)

                time_list[ii]=time_matrix
                state_list[ii]=state_matrix


            #print(ii)

        self.time_list=time_list
        self.state_list=state_list
        self.effect_matrix=effect_matrix
        self.big_number_matrix=big_number_matrix
        self.dis_matrix=dis_matrix


        return self.time_list,self.state_list,self.effect_matrix, self.big_number_matrix, self.dis_matrix

    def GLS_s(self, t=0.1, repeat=2, ini=None, end=None, ifrecal=True):

        if ifrecal==True:
            self.GLS_m(t=t, ini=ini, end=end, repeat = repeat)

        time_list=self.time_list
        state_list=self.state_list
        effect_matrix=self.effect_matrix
        big_number_matrix=self.big_number_matrix
        dis_matrix=self.dis_matrix

        max_number = 10
        time_matrix = 100*np.ones(shape=(self.sites_length, max_number))
        state_matrix = np.zeros(shape=(self.sites_length, max_number))
        effect_number=0
        big_number=0

        for i in range(self.sites_length):
            a = np.random.choice(range(repeat), 1, p=(1 / float(repeat)) * np.ones(repeat))[0]
            if big_number_matrix[i,a]<=max_number:

                    time_matrix[i, 0:max_number] = time_list[i][a,]
                    state_matrix[i, 0:max_number] = state_list[i][a,]
                    big_number=max(big_number_matrix[i,a],big_number)
                    effect_number=effect_number+effect_matrix[i,a]
            else:
                    big_number=max(big_number_matrix[i,a],big_number)
                    effect_number=effect_number+effect_matrix[i,a]

                    time_matrix_old = time_matrix
                    state_matrix_old = state_matrix
                    time_matrix = np.zeros(shape=(self.sites_length, big_number))
                    state_matrix = np.zeros(shape=(self.sites_length, big_number))
                    time_matrix[0:self.sites_length, 0:max_number] = time_matrix_old
                    state_matrix[0:self.sites_length, 0:max_number] = state_matrix_old
                    time_matrix[i, 0:big_number] = time_list[i][a,]
                    state_matrix[i, 0:big_number] = state_list[i][a,]

        time_matrix = time_matrix[0:self.sites_length, 0:int(big_number)]
        state_matrix = state_matrix[0:self.sites_length, 0:int(big_number)]

        return time_matrix, state_matrix, int(effect_number), int(big_number)




    def whether_IGC(self,history_matrix,effect_number):

        p_h=np.zeros(shape=(effect_number, 4))

# 0 difference, 1 time, 2 whether igc

        if self.Model == "HKY":
            for ii in range(effect_number):
                p_h[ii,0]=history_matrix[ii,1]
                p_h[ii, 1] = history_matrix[ii, 4]

                i_b=int(history_matrix[ii, 6])//4
                j_b = int(history_matrix[ii, 6]) % 4
                i_p = int(history_matrix[ii, 7]) // 4
                j_p = int(history_matrix[ii, 7]) % 4

                if i_p == j_p:
                    if i_b != j_b and i_b == i_p:
                        # y_coor is corresponding coor for igc
                        y_coor = np.argwhere(self.dic_col[int(history_matrix[ii, 6]),] == (int(history_matrix[ii, 7]) + 1))[0]
                        qq=self.Q[int(history_matrix[ii, 6]), y_coor]
                        u=random.uniform(0,1)
                        if u <= float((np.exp(self.tau))/qq):
                            p_h[ii, 2] = 1

                    elif(i_b!=j_b and j_b==j_p):
                        y_coor=np.argwhere(self.dic_col[int(history_matrix[ii, 6]),] == (int(history_matrix[ii, 7]) + 1))[0]
                        qq=self.Q[int(history_matrix[ii, 6]),y_coor]
                        u=random.uniform(0,1)
                        if u<=float((np.exp(self.tau))/qq):
                            p_h[ii, 2] =1

        else:
            for ii in range(effect_number):
                p_h[ii, 0] = history_matrix[ii, 1]
                p_h[ii, 1] = history_matrix[ii, 4]

                i_b = int(history_matrix[ii, 6]) // 61
                j_b = int(history_matrix[ii, 6]) % 61
                i_p = int(history_matrix[ii, 7]) // 61
                j_p = int(history_matrix[ii, 7]) % 61

                if (i_p == j_p):
                    if (i_b != j_b and i_b == i_p):
                        y_coor = np.argwhere(self.dic_col[int(history_matrix[ii, 6]),] == (int(history_matrix[ii, 7]) + 1))[0]
                        qq = self.Q[int(history_matrix[ii, 6]), y_coor]
                        u = random.uniform(0, 1)

                        ca = geneconv.state_to_codon[j_b]
                        ca=self.codon_table[ca]
                        cb = geneconv.state_to_codon[j_p]
                        cb=self.codon_table[cb]

                        if isNonsynonymous(cb, ca, self.codon_table):
                            tau = self.tau * self.omega
                        else:
                            tau = self.tau

                        if (u <= (float(tau) / qq)):
                            p_h[ii, 2] = 1

                    elif (i_b != j_b and j_b==j_p):
                        y_coor = np.argwhere(self.dic_col[int(history_matrix[ii, 6]),] == (int(history_matrix[ii, 7]) + 1))[0]
                        qq = self.Q[int(history_matrix[ii, 6]), y_coor]
                        u = random.uniform(0, 1)

                        ca = geneconv.state_to_codon[i_b]
                        ca = self.codon_table[ca]
                        cb = geneconv.state_to_codon[i_p]
                        cb = self.codon_table[cb]

                        if isNonsynonymous(cb, ca, self.codon_table):
                            tau = self.tau * self.omega
                        else:
                            tau = self.tau
                        if (u <= (float(tau)/qq)):
                            p_h[ii, 2] = 1

        for ii in range(effect_number-1):
            if p_h[ii,1]==0:
                p_h=p_h[0:ii-1,0:4]
                effect_number=ii-1
                break


        return p_h, effect_number


    def rank_ts(self,time,state,ini,effect_number):


        if self.dic_di is None:
            self.making_dic_di()

        di=self.dic_di

        difference=0
        time_new=0


        for i in range(self.sites_length):
            difference = difference+di[ini[i]]

    # 0 last difference number ,1 next difference number, 2 last time, 3 next time
    # 4 time difference is important, 5 location ,6 last state, 7 next state,8 paralog type
        history_matrix = np.zeros(shape=(effect_number+1, 8))
        for jj in range(effect_number+1):
            coor = np.argmin(time)
            history_matrix[jj,0]=difference
            time_old=time_new
            history_matrix[jj, 2] = time_old
            time_new=np.min(time)
            history_matrix[jj, 3] = time_new
            history_matrix[jj, 4] = time_new-time_old
    # track state matrix
            d = time.shape[1]
            x_aixs = coor / d
            y_aixs = coor % d
            history_matrix[jj, 5]=x_aixs
            history_matrix[jj, 6] = ini[x_aixs]
            history_matrix[jj, 7]=state[x_aixs, y_aixs]

            history_matrix[jj, 1]=difference-di[int(history_matrix[jj, 6])]+di[int(history_matrix[jj, 7])]
            difference=history_matrix[jj, 1]
            # renew time matrix and ini stat
            time[x_aixs, y_aixs]=100
            ini[x_aixs]=history_matrix[jj, 7]

        return history_matrix,effect_number



    def monte_carol(self,times=60,repeat=15,ifwholetree=False):

        if ifwholetree == False:
            ini1 = self.make_ie(2, 3)[0]
            end1 = self.make_ie(2, 3)[1]
            re = self.GLS_s(repeat=repeat,ini=ini1,end=end1)

            sam = self.rank_ts(time=re[0], state=re[1], ini=ini1, effect_number=re[2])
            re1=self.whether_IGC(history_matrix=sam[0],effect_number=sam[1])
            effect_number=re1[1]
            re1=re1[0]

            for i in range(times-1):
                re = self.GLS_s(ifrecal=False,ini=ini1,end=end1)
                sam = self.rank_ts(time=re[0], state=re[1], ini=ini1, effect_number=re[2])
                re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1])
                re1=np.vstack((re1,re2[0]))
                effect_number=effect_number+re2[1]


        else:
            for j in range(len(scene['tree']["column_nodes"])):
                if j ==0:
                    ini2=self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                    end2 = 2
                    ini1 = self.make_ie(ini2, end2)[0]
                    end1 = self.make_ie(ini2, end2)[1]
                    re = self.GLS_s(repeat=repeat,ini=ini1,end=end1)
                    sam = self.rank_ts(time=re[0], state=re[1], ini=ini1, effect_number=re[2])
                    re1=self.whether_IGC(history_matrix=sam[0],effect_number=sam[1])
                    effect_number=re1[1]
                    re1=re1[0]


                    for i in range(times-1):
                        re = self.GLS_s(ifrecal=False,ini=ini1,end=end1)
                        sam = self.rank_ts(time=re[0], state=re[1], ini=ini1, effect_number=re[2])
                        re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1])
                        re1=np.vstack((re1,re2[0]))
                        effect_number=effect_number+re2[1]



                elif j==1:
                    print("Ingore the outgroup")


                else:
                    ini2=self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                    end2 = self.geneconv.node_to_num[geneconv.edge_list[j][1]]

                    ini1 = self.make_ie(ini2, end2)[0]
                    end1 = self.make_ie(ini2, end2)[1]
                    re = self.GLS_s(repeat=repeat,ini=ini1,end=end1)

                    sam = self.rank_ts(time=re[0], state=re[1], ini=ini1, effect_number=re[2])
                    re2=self.whether_IGC(history_matrix=sam[0],effect_number=sam[1])
                    effect_number1=re2[1]
                    re2=re2[0]

                    for i in range(times-1):
                        re = self.GLS_s(ifrecal=False,ini=ini1,end=end1)
                        sam = self.rank_ts(time=re[0], state=re[1], ini=ini1, effect_number=re[2])
                        re3 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1])
                        re2=np.vstack((re2,re3[0]))
                        effect_number1=effect_number1+re3[1]

            re1 = np.vstack((re1, re2))
            effect_number=effect_number1+effect_number

        return re1,effect_number


    def divide_Q(self, times=1, repeat=1,method="simple", ifwholetree=True,simple_state_number=8):

        re=self.monte_carol(times=times,repeat=repeat,ifwholetree=ifwholetree)
        history_matrix=re[0]
        effect_number=re[1]
        if (method == "simple"):
            quan = 1 / float(simple_state_number)
            quan_c = 1 / float(simple_state_number)
            stat_rank = pd.DataFrame(history_matrix[:, 0])
            stat_vec = np.zeros(shape=(simple_state_number - 1, 1))
            for i in range(simple_state_number - 1):
                stat_vec[i] = np.quantile(stat_rank, quan_c)
                quan_c = quan_c + quan

            print(stat_vec)
            for ii in range(effect_number ):
                if (history_matrix[ii, 0] <= stat_vec[0]):
                    history_matrix[ii, 3] = 0
                elif (history_matrix[ii, 0] > stat_vec[simple_state_number - 2]):
                    history_matrix[ii, 3] = simple_state_number - 1
                else:
                    for iii in range(simple_state_number - 1):
                        if (history_matrix[ii, 0] <= stat_vec[iii + 1] and history_matrix[ii, 0] > stat_vec[iii]):
                            history_matrix[ii, 3] = iii + 1
                            break

            type_number = simple_state_number

        self.igc_com=history_matrix
        self.type_number=int(type_number)
        self.last_effct=int(effect_number)

        return history_matrix



    def get_igcr_pad(self,times=1, repeat=1,simple_state_number=8,ifwholetree=True):
        self.divide_Q(times=times,repeat=repeat,simple_state_number=simple_state_number,ifwholetree=ifwholetree)
        relationship=np.zeros(shape=(self.type_number, 5))
        for i in range(self.type_number):
            for j in range(self.last_effct):
                if(self.igc_com[j,3]==i):
                    relationship[i,0]= self.igc_com[j,2]+relationship[i,0]
                    relationship[i, 1] = self.igc_com[j, 1]+relationship[i, 1]
                    relationship[i, 2] = self.igc_com[j, 0]+relationship[i, 2]
                    relationship[i,4]=1+relationship[i,4]

                relationship[i, 2]=float(relationship[i,2])
                relationship[i, 3] = float(relationship[i, 0]) / (relationship[i, 1])

        return relationship


if __name__ == '__main__':


    paralog = ['EDN', 'ECP']
    alignment_file = '../test/EDN_ECP_Cleaned.fasta'
    newicktree = '../test/EDN_ECP_tree.newick'
    Force = {5:1}
    Force1=None
    model = 'HKY'


    name='EDN_ECP_full'
    type='situation1'
    save_name = '../test/save/' + model + name+'_'+type+'_nonclock_save.txt'
    geneconv = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name)

    test = AncestralState(geneconv)
    self = test

    scene = self.get_scene()



    #print(self.get_igcr_pad())

    # print(self.node_length)

    # print(geneconv.edge_to_blen)
    # print(geneconv.num_to_node)
    # print(geneconv.edge_list)
    # print(scene['tree'])
    #
    # for i in range(len(scene['tree']["column_nodes"])):
    #     print(scene['tree']["column_nodes"][i])
    #     print(geneconv.edge_list[i])
    #
    #
    #self.rank_ts()
       # self.monte_carol(times=2,repeat=2)
        # print(self.igc_com)

    #aa=self.monte_carol(times=1,repeat=2)



    #
    # save=self.get_maxpro_matrix(True,1)
    # save_namep = '../test/savecommon32/Ind_' + model + '_1_'+type+'_'+name+'_maxpro.txt'
    # np.savetxt(open(save_namep, 'w+'), save.T)
    #
    #
    # save=self.get_maxpro_matrix(True,2)
    # save_namep = '../test/savecommon32/Ind_' + model + '_2_'+type+'_'+name+'_maxpro.txt'
    # np.savetxt(open(save_namep, 'w+'), save.T)
    #
    # save=self.get_maxpro_index(True,1)
    # save_namep = '../test/savecommon32/Ind_' + model + '_1_'+type+'_'+name+'_maxproind.txt'
    # np.savetxt(open(save_namep, 'w+'), save.T)
    #
    # save=self.get_maxpro_index(True,2)
    # save_namep = '../test/savecommon32/Ind_' + model + '_2_'+type+'_'+name+'_maxproind.txt'
    # np.savetxt(open(save_namep, 'w+'), save.T)

    #
    # interior_node=self.get_interior_node()
    # for i in interior_node:
    #     for j in range(2):
    #         mar=self.get_marginal(node=i,paralog=(j+1))
    #         ii=str(i)
    #         jj=str(j+1)
    #         save_namem = '../test/savecommon3/Ind_' + model +"_node"+ii+"_paralog"+jj+ "_"+name+'_mag.txt'
    #         np.savetxt(open(save_namem, 'w+'), mar.T)

    # save_name = '../test/save/' + model + '2_Force_YBR191W_YPL079W_nonclock_save.txt'
    # geneconv = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
    #                            save_path='../test/save/', save_name=save_name)
    # test = AncestralState(geneconv)
    # self = test
    # scene = self.get_scene()


    # site_num = 0
    # save=self.get_marginal(1)

    # print(scene)
#     save=self.get_maxpro_matrix(True,1)
#     save_namep = '../test/savecommon3/Ind_' + model + '_1_Force_'+name+'_maxpro.txt'
#     np.savetxt(open(save_namep, 'w+'), save.T)
#
#     save=self.get_maxpro_matrix(True,2)
#     save_namep = '../test/savecommon3/Ind_' + model + '_2_Force_'+name+'_maxpro.txt'
#     np.savetxt(open(save_namep, 'w+'), save.T)
#
#     save=self.get_maxpro_index(True,1)
#     save_namep = '../test/savecommon3/Ind_' + model + '_1_Force_'+name+'_maxproind.txt'
#     np.savetxt(open(save_namep, 'w+'), save.T)
#
#     save=self.get_maxpro_index(True,2)
#     save_namep = '../test/savecommon3/Ind_' + model + '_2_Force_'+name+'_maxproind.txt'
#     np.savetxt(open(save_namep, 'w+'), save.T)
#
#
#     interior_node=self.get_interior_node()
#     for i in interior_node:
#         for j in range(2):
#             mar=self.get_marginal(node=i,paralog=(j+1))
#             ii=str(i)
#             jj=str(j+1)
#             save_namem = '../test/savecommon3/Ind_' + model +"_node"+ii+"_paralog"+jj+ '_Force_'+name+'mag.txt'
#             np.savetxt(open(save_namem, 'w+'), mar.T)
#
#
# ##    aa = 0
    # for i in range(len(j_out["responses"][0][0])):
    #     print(j_out["responses"][0][0][i]1)
    #     aa=array(j_out["responses"][0][0][i])+aa
    # aa=self.get_scene()
    # print(aa["observed_data"])
##    re = self.get_scene()
##    list_for_iid = re["observed_data"]["iid_observations"]
##    list_commonan = []
##    for i in range(len(list_for_iid)):
##    # for i in range(3):
##        re["observed_data"]["iid_observations"] = [list_for_iid[i]]
##
##        requests = [
##            {"property": "DNDNODE"}
##        ]
##        j_in = {
##            'scene': re,
##            'requests': requests
##        }
##        j_out = jsonctmctree.interface.process_json_in(j_in)
##        j_out_matrix = np.array(j_out["responses"][0][0])
##        list_commonan.append(j_out_matrix)
##        # print(re["observed_data"]["iid_observations"])
##    #  print(aa["process_definitions"][0]["row_states"])
##    #  print(aa["process_definitions"][0]["column_states"])
##    #  print(aa["process_definitions"][0]["transition_rates"])
##    list_node=get_interior_node(re)
##    dict=self.get_dict_trans()
##    len_node=len(list_node)
##    len_se=len(list_commonan)
##    get_maxpro=get_maxpro(list_commonan,list_node)
##    # print(get_maxpro[2][2]%61)
    ##    translate=translate_into_seq(promax=get_maxpro,len_node=len_node,dict=dict,model=model,len_se=len_se)


