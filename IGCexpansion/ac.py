# coding=utf-8
# A separate file for Ancestral State Reconstruction
# Uses Alex Griffing's JsonCTMCTree package for likelihood and gradient calculation
# Xiang Ji
# xji4@tulane.edu
# Tanchumin Xu
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
        self.tau =geneconv.tau
        self.omega=geneconv.omega
        self.Q_new = None
        self.Q= None

        self.sites1 = None
        self.sites2 = None
        self.sites=None
        self.dic_di = None
        self.tauoriginal=0

        # history from monte carlo
        self.time_list= None
        self.state_list = None
        self.effect_matrix= None
        self.big_number_matrix= None
        self.dis_matrix = None

        self.node_length=0
        self.sites_length = self.geneconv.nsites
        self.Model=self.geneconv.Model
        self.ifmarginal = False

        self.min_diff=0

# relationship is a matrix about igc rates on different paralog
# igc_com is matrix contain paralog difference,difference time, igc state, paralog category
        self.relationship=None
        self.igc_com=None

        self.judge=None
        self.P_list= None
        self.Q_original=None
        self.ifmax=False
        self.ifXiang=False

    def get_mle(self):
        self.geneconv.get_mle()

    def get_scene(self):
        if self.scene is None:
            self.get_mle()
            self.scene = self.geneconv.get_scene()
        return self.scene

    def get_dict_trans(self):
        return self.geneconv.get_dict_trans()

    def get_ancestral_state_response(self,iffix=False):
        self.get_scene()

        if iffix==True:


            for i in range(len(self.scene['tree']["column_nodes"])):
                if i ==1:
                    self.scene['tree']["edge_rate_scaling_factors"][i]=0.1
                else:
                    self.scene['tree']["edge_rate_scaling_factors"][i]=0.02

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

    def get_ancestral_state_response_x(self):
        self.get_scene()

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


#get original common ancestral matrix
    def get_maxpro_index(self):

        self.ancestral_state_response = deepcopy(self.get_ancestral_state_response())

        if self.ifmax == False:
            self.node_length=len(self.get_num_to_node())
            sites = np.zeros(shape=(self.node_length,self.sites_length ))
            for site in range(self.sites_length):
                for node in range(self.node_length):
                    if self.Model=="HKY":
                        # print(np.array(self.ancestral_state_response[site])[:, node])
                         sites[node][site] =np.random.choice(range(16), 1, p=np.array(self.ancestral_state_response[site])[:, node])[0]
                        # print(sites[node][site])
            self.sites=sites

        else:
            self.node_length=len(self.get_num_to_node())
            sites = np.zeros(shape=(self.node_length,self.sites_length ))
            for site in range(self.sites_length):
                for node in range(self.node_length):
                    if self.Model=="HKY":
                        # print(np.array(self.ancestral_state_response[site])[:, node])
                         sites[node][site] =np.argmax(np.array(self.ancestral_state_response[site])[:, node])
                        # print(sites[node][site])
            self.sites=sites



        return (sites)

    def get_maxpro_matrix(self):

        if self.ancestral_state_response is None:
            self.ancestral_state_response = self.get_ancestral_state_response()

        self.node_length=len(self.get_num_to_node())
        sites = np.zeros(shape=(self.node_length, self.sites_length))
        for site in range(self.sites_length):
            for node in range(self.node_length):
                sites[node][site] = np.max(np.array(self.ancestral_state_response[site])[:, node])

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


#get the index for  internal node
    def get_interior_node(self):

        if self.node_length ==0:
             self.node_length = len(self.scene['tree']["column_nodes"])

        node = np.arange(self.node_length)
        interior_node = set(node) - set(self.scene["observed_data"]["nodes"])
     #   print(self.scene["observed_data"]["iid_observations"])
     #   print(self.scene["observed_data"]["variables"])
     #   print(self.scene["observed_data"]["nodes"])
        c = [i for i in interior_node]

        return (c)


#classfy states by how many different leaves it  has
    def judge_state_children(self):
        internal_node=self.get_interior_node()
        judge=np.ones(int((len(self.scene['tree']["column_nodes"])+1)/2))

        for i in range(int((len(self.scene['tree']["column_nodes"])+1)/2)):

            end1=self.geneconv.node_to_num[geneconv.edge_list[i*2][1]]
            end2 = self.geneconv.node_to_num[geneconv.edge_list[(i * 2)+1][1]]

            if(end1 in set(internal_node) ):
                judge[i] =3
            elif (end2 in set(internal_node) ):
                judge[i] = 2

            elif(end1 in set(internal_node) and end2 in set(internal_node)):
                judge[i] = 1
            else:
                judge[i] = 0

        self.judge=judge

       # self.get_scene()
      #  scene=self.scene
      #  print(scene['process_definitions'])

# compute Q matrix  with 0
    def original_Q(selfs):
        if self.Q is None:
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

# making matrices to store possibility for internal node
    def making_possibility_internal(self):

        if self.judge is None:
            self.judge_state_children()

        # self.Q original is 16*16 Q matrix
        if self.Q_original is None:
            self.original_Q()

        self.get_maxpro_index()

        internal_node = self.get_interior_node()
        P_list = []
        tree_len=len(self.scene['tree']["column_nodes"])

# P_list store the P matrix
        if self.Model=="HKY":
            statenumber = 16
            for i in range(tree_len):
                if i==1:
                    time = self.scene['tree']["edge_rate_scaling_factors"][i]
                    Q1 = geneconv.get_HKYBasic()

                    for k in range(4):
                          Q1[k, k] = -sum(Q1[k,])
                    Q=linalg.expm(Q1 * time)
                    P_list.append(Q)
                else:
                    time=self.scene['tree']["edge_rate_scaling_factors"][i]
                    Q=linalg.expm(self.Q_original * time)
                    P_list.append(Q)

            save_nameP = '../test/savesample/Ind_' + "QQQ" + 'sample.txt'
            np.savetxt(save_nameP, self.Q_original.T)

        else:
            statenumber = 3721
            for i in range(tree_len):
                if i == 1:
                    time = self.scene['tree']["edge_rate_scaling_factors"][i]
                    Q1 = geneconv.get_MG94Basic()
                    for k in range(61):
                        Q1[k, k] = -sum(Q1[k,])
                    Q = linalg.expm(Q1 * time)
                    P_list.append(Q)
                else:
                    time = self.scene['tree']["edge_rate_scaling_factors"][i]
                    Q = linalg.expm(self.Q_original * time)
                    P_list.append(Q)

            save_nameP = '../test/savesample/Ind_' + "QQQ" + 'sample.txt'
            np.savetxt(save_nameP, self.Q_original.T)



        #p_n is list store the  probabolity matrices for  internal nodel
        p_n=[]
        for i in range(len(self.judge)):
            p_n.append(0)

        # tree_to store the topology of each internal  node
        tree_to=np.zeros(shape=(3, len(self.judge)))


        for i in range(len(self.judge)-1):
            inode= internal_node[len(self.judge)-1-i]
            state=int(self.judge[len(self.judge)-1-i])
            id=1
            for j in range(tree_len-1):
                if(self.geneconv.node_to_num[geneconv.edge_list[j][0]]==inode and id==1):
                    id=id+1
                    left = self.geneconv.node_to_num[geneconv.edge_list[j][1]]
                    right= self.geneconv.node_to_num[geneconv.edge_list[j+1][1]]
                    tree_to[1,len(self.judge)-i-1]=left
                    tree_to[2,len(self.judge)-i-1]=right
                    p_node = np.zeros(shape=(self.sites_length, statenumber))

                    if(state==0):
                        for sites in range(self.sites_length):
                            leftpoint=int(self.sites[left,sites])
                            rightpoint=int(self.sites[right,sites])
                            for current in range(statenumber):
                                 p_node[sites,current]=P_list[j+1][current,rightpoint]*P_list[j][current,leftpoint]

                        p_n[len(self.judge)-i-1]=p_node

                    elif(state==2):
                        right = int(internal_node.index(right))
                        rightm = p_n[right]
                        # rightm is p we caulcate before which is a matrix site.length*16
                        for kk in range(self.sites_length):
                            leftpoint = int(self.sites[left, kk])
                            rightpoint=rightm[kk,]
                            for current in range(statenumber):
                                p1=0
                                for kkkk in range(statenumber):
                                    p1=P_list[j+1][current,kkkk]*rightpoint[kkkk]+p1
                                p_node[kk, current] = p1* P_list[j][current,leftpoint]
                        p_n[len(self.judge) - i-1] = p_node


                    elif(state==3):
                        left = int(internal_node.index(left))
                        leftm = p_n[left]
                        # leftm is p we caulcate before which is a matrix site.length*16
                        for sites in range(self.sites_length):
                            rightpoint = int(self.sites[right, sites])
                            leftpoint=leftm[sites,]
                            for current in range(statenumber):
                                p1=0
                                for kkkk in range(statenumber):
                                    p1=P_list[j][current,kkkk]*leftpoint[kkkk]+p1
                                p_node[sites, current] = p1* P_list[j+1][current,rightpoint]
                        p_n[len(self.judge) - i-1] = p_node

                    else:
                        left = int(internal_node.index(left))
                        leftm = p_n[left]
                        right = int(internal_node.index(right))
                        rightm = p_n[right]

                        # leftm is p we calculate before which is a matrix site.length*16
                        for kk in range(self.sites_length):
                            leftpoint=leftm[kk,]
                            rightpoint = rightm[kk,]
                            for kkk in range(statenumber):
                                p1=0
                                p2=0
                                for kkkk in range(statenumber):
                                    p1=P_list[j][kkk,kkkk]*leftpoint[kkkk]+p1
                                    p2 = P_list[j+1][kkk, kkkk] * rightpoint[kkkk] + p2
                                p_node[kk, kkk] = p1* p2
                        p_n[len(self.judge) - i-1] = p_node

        self.p_n=p_n
        self.P_list=P_list
        self.tree_to=tree_to

       # print((self.sites[2,1]%4))



    def jointly_common_ancstral_inference(self,ifcircle=False,taulist=None):

        if self.ifXiang==False:
            if ifcircle==False:
                self.making_possibility_internal()
            else:
                self.making_possibility_internal_EM(taulist=taulist)

            tree_len = len(self.scene['tree']["column_nodes"])
            internal_node = self.get_interior_node()
            index=1
            j=0
           # print(internal_node)

    # try to find parent of internal node
            while index < len(internal_node):
                if (self.geneconv.node_to_num[geneconv.edge_list[j][1]]==internal_node[index]):
                    self.tree_to[0,index]=self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                    index=index+1
                j=j+1


            if self.Model=="HKY":
                for j in range(self.sites_length):
                    self.sites[0,j]=np.random.choice(range(16), 1, p=np.array(self.ancestral_state_response[j])[:, 0])[0]

                for i in range(tree_len):
                    if(i>0 and i in set(internal_node)):
                        index=internal_node.index(i)
                        for j in range(self.sites_length):
                            selectp=np.ones(16)
                            parent = int(self.tree_to[0, (index)])
                           # print(parent)
                            parent = int(self.sites[parent,j])
                            for k in range(16):
                                selectp[k]=self.P_list[i-1][parent,k]*self.p_n[index][j,k]
                            selectp=selectp/sum(selectp)
                            if(np.any(selectp)<0):
                                print(selectp)
                            self.sites[i,j]=np.random.choice(range(16), 1, p=selectp)[0]



            else:
                for j in range(self.sites_length):
                    self.sites[0, j] = \
                    np.random.choice(range(3721), 1, p=np.array(self.ancestral_state_response[j])[:, 0])[0]

                for i in range(tree_len):
                    if (i > 0 and i in set(internal_node)):
                        index = internal_node.index(i)
                        for j in range(self.sites_length):
                            selectp = np.ones(3721)
                            parent = int(self.tree_to[0, (index)])
                            # print(parent)
                            parent = int(self.sites[parent, j])
                            for k in range(3721):
                                selectp[k] = self.P_list[i - 1][parent, k] * self.p_n[index][j, k]
                            selectp = selectp / sum(selectp)
                            self.sites[i, j] = np.random.choice(range(3721), 1, p=selectp)[0]




        if  self.ifXiang==True:
            self.ancestral_state_response = deepcopy(self.get_ancestral_state_response_x())

            self.node_length = len(self.get_num_to_node())
            sites = np.zeros(shape=(self.node_length, self.sites_length))
            for site in range(self.sites_length):
                for node in range(self.node_length):
                    sites[node][site] = int(np.array(self.ancestral_state_response[site])[node])

            self.sites = sites

    def making_possibility_internal_EM(self,taulist=None):

            if self.judge is None:
                self.judge_state_children()

            ifsimulate=True


            self.get_maxpro_index()

            internal_node = self.get_interior_node()
            P_list = []
            tree_len = len(self.scene['tree']["column_nodes"])

            # P_list store the P matrix
            if self.Model=="HKY":
                statenumber = 16
                for i in range(tree_len):
                    if i == 1:
                        time = self.scene['tree']["edge_rate_scaling_factors"][i]
                        Q1 = geneconv.get_HKYBasic()
                        for k in range(4):
                            Q1[k, k] = -sum(Q1[k,])
                        Q = linalg.expm(Q1 * time)
                        P_list.append(Q)
                    else:
                        time = self.scene['tree']["edge_rate_scaling_factors"][i]
                        if(i==0):
                            if ifsimulate == True:
                                self.change_t_Q(tau=1)
                                self.original_Q()
                        if(i>=2):
                            self.change_t_Q(tau=taulist[i-2])
                            self.original_Q()
                        Q = linalg.expm(self.Q_original * time)
                        P_list.append(Q)
            else:
                statenumber = 3721
                for i in range(tree_len):
                    if i == 1:
                        time = self.scene['tree']["edge_rate_scaling_factors"][i]
                        Q1 = geneconv.get_MG94Basic()
                        for k in range(3721):
                            Q1[k, k] = -sum(Q1[k,])
                        Q = linalg.expm(Q1 * time)
                        P_list.append(Q)
                    else:
                        time = self.scene['tree']["edge_rate_scaling_factors"][i]
                        if(i==0):
                            if ifsimulate == True:
                                self.change_t_Q(tau=1)
                                self.original_Q()
                        if(i>=2):
                            self.change_t_Q(tau=taulist[i-2])
                            self.original_Q()
                        Q = linalg.expm(self.Q_original * time)
                        P_list.append(Q)


            # p_n is list store the  probabolity matrices for  internal nodel
            p_n = []
            for i in range(len(self.judge)):
                p_n.append(0)

            # tree_to store the topology of each internal  node
            tree_to = np.zeros(shape=(3, len(self.judge)))

            for i in range(len(self.judge) - 1):
                inode = internal_node[len(self.judge) - 1 - i]
                state = int(self.judge[len(self.judge) - 1 - i])
                id = 1
                for j in range(tree_len - 1):
                    if (self.geneconv.node_to_num[geneconv.edge_list[j][0]] == inode and id == 1):
                        id = id + 1
                        left = self.geneconv.node_to_num[geneconv.edge_list[j][1]]
                        right = self.geneconv.node_to_num[geneconv.edge_list[j + 1][1]]
                        tree_to[1, len(self.judge) - i - 1] = left
                        tree_to[2, len(self.judge) - i - 1] = right
                        p_node = np.zeros(shape=(self.sites_length, statenumber))

                        if (state == 0):
                            for sites in range(self.sites_length):
                                leftpoint = int(self.sites[left, sites])
                                rightpoint = int(self.sites[right, sites])
                                for current in range(statenumber):
                                    p_node[sites, current] = P_list[j + 1][current, rightpoint] * P_list[j][
                                        current, leftpoint]

                            p_n[len(self.judge) - i - 1] = p_node

                        elif (state == 2):
                            right = int(internal_node.index(right))
                            rightm = p_n[right]
                            # rightm is p we caulcate before which is a matrix site.length*16
                            for kk in range(self.sites_length):
                                leftpoint = int(self.sites[left, kk])
                                rightpoint = rightm[kk,]
                                for current in range(statenumber):
                                    p1 = 0
                                    for kkkk in range(statenumber):
                                        p1 = P_list[j + 1][current, kkkk] * rightpoint[kkkk] + p1
                                    p_node[kk, current] = p1 * P_list[j][current, leftpoint]
                            p_n[len(self.judge) - i - 1] = p_node


                        elif (state == 3):
                            left = int(internal_node.index(left))
                            leftm = p_n[left]
                            # leftm is p we caulcate before which is a matrix site.length*16
                            for sites in range(self.sites_length):
                                rightpoint = int(self.sites[right, sites])
                                leftpoint = leftm[sites,]
                                for current in range(statenumber):
                                    p1 = 0
                                    for kkkk in range(statenumber):
                                        p1 = P_list[j][current, kkkk] * leftpoint[kkkk] + p1
                                    p_node[sites, current] = p1 * P_list[j + 1][current, rightpoint]
                            p_n[len(self.judge) - i - 1] = p_node

                        else:
                            left = int(internal_node.index(left))
                            leftm = p_n[left]
                            right = int(internal_node.index(right))
                            rightm = p_n[right]
                            # leftm is p we calculate before which is a matrix site.length*16
                            for kk in range(self.sites_length):
                                leftpoint = leftm[kk,]
                                rightpoint = rightm[kk,]
                                for kkk in range(statenumber):
                                    p1 = 0
                                    p2 = 0
                                    for kkkk in range(statenumber):
                                        p1 = P_list[j][kkk, kkkk] * leftpoint[kkkk] + p1
                                        p2 = P_list[j + 1][kkk, kkkk] * rightpoint[kkkk] + p2
                                    p_node[kk, kkk] = p1 * p2
                            p_n[len(self.judge) - i - 1] = p_node

            self.p_n = p_n
            self.P_list = P_list
            self.tree_to = tree_to

           # # P_list
           #  state16 = np.array([0, 5, 10, 15])
           #  pro = np.array([0, 5, 10, 15], dtype=float)
           #
           #  for i in range(self.sites_length):
           #      for j in range(4):
           #          pp = 0
           #          for kk in range(16):
           #              pp=pp+(p_n[1][i,kk]*P_list[0][state16[j],kk])
           #
           #          pp1=int(self.sites[2, i] % 4)
           #          pro[j] = self.geneconv.pi[j] *pp * P_list[1][j,pp1]
           #      selectp = pro / (np.sum(pro))
           #
           #      self.sites[0, i] = int(np.random.choice(range(4), 1, p=selectp)[0])
           #      state4=self.sites[0,i]
           #      self.sites[0, i] = state4 * 4 + state4
           #    #  print(self.sites[0, i])





# test common ancster inferece
    def test_pro(self,node=1,sites=1,to=[1,2,3],leaf=[4,6,7,8]):

        if self.P_list is None:
            self.making_possibility_internal()

        tree_len = len(self.scene['tree']["column_nodes"])
        internal_node = self.get_interior_node()
        index=1
        j=0

        print(internal_node)


        while index < len(internal_node):
            if (self.geneconv.node_to_num[geneconv.edge_list[j][1]]==internal_node[index]):
                self.tree_to[0,index]=self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                index=index+1
            j=j+1

        for i in  range(tree_len):
            if(i== node):
                index=internal_node.index(i)
                j=sites
                selectp=np.ones(16)
                for k in range(16):
                    selectp[k]=self.p_n[index][j,k]
                selectp = selectp / sum(selectp)

        list=[]
        for j in range(len(to)):
            time=0.2
            Q=linalg.expm(self.Q_original * time*to[j])

            list.append(Q)

        nodesite=np.ones(16)
        for j in range(len(leaf)):
            leafsite=self.sites[leaf[j],sites]
            if(j <=1):
                for k in range(16):
                     nodesite[k]=list[j][k,int(leafsite)]*nodesite[k]


            else:
                for k in range(16):
                     nodesite[k]=list[2][k,int(leafsite)]*nodesite[k]

        nodesite = nodesite/sum(nodesite)


        print(nodesite-selectp)

    def test_pro11(self,node_s=[0,1,1,1],site_s=1,mc=10):

        if self.P_list is None:
            self.making_possibility_internal()


        self.jointly_common_ancstral_inference()

        tree_len = len(self.scene['tree']["column_nodes"])
        internal_node = self.get_interior_node()

        sites_new=self.sites

        for i in  range(4):
            sites_new[internal_node[i]][site_s]=node_s[i]

        p1=0

        sites_test=np.array(self.get_ancestral_state_response_x()[site_s])
        for i in range(len(internal_node)):
            sites_test[internal_node[i]]=node_s[i]
        print(sites_test)

        self.scene['observed_data']["iid_observations"]=[self.scene['observed_data']["iid_observations"][site_s]]

    #    print(np.array(self.ancestral_state_response[1])[:, 6])


        for mctimes in range(mc):


            sites_mc=self.get_ancestral_state_response_x()[0]
           #  print(mctimes)
            print(sites_mc)
            if((sites_mc==sites_test).all()):
                 p1=p1+1

        print(p1/mc)


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

# making Q matrix by dividing diagnal
    def making_Qg(self):

        if self.Q is None:
           self.making_Qmatrix()

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


        Q_iiii = np.ones((di))
        for ii in range(di):
            Q_iiii[ii] = sum(self.Q[ii,])

        for d in range(di):
            self.Q_new[d,] = self.Q[d,] / Q_iiii[d]

        return self.Q_new

    # making Q matrix
    def making_Qmatrix(self):

        self.get_scene()
        scene=self.scene

        actual_number=(len(scene['process_definitions'][1]['transition_rates']))
        self.actual_number=actual_number

        global x_i
        x_i = 0

        global index
        index = 0

# dic is from 1:16

        if self.Model == 'HKY':
            self.Q=np.zeros(shape=(16, 9))
            self.dic_col=np.zeros(shape=(16, 9))
            for i in range(actual_number):

# x_io means current index for row states, x_i is states for last times
# self.dic_col indicates the coordinates for ending states

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



# end point condition to generate history
# 100 is the largest time interval
    def  GLS_m(self,t=0.1, ini=None, end=None, repeat=1,if_circle=False):

        global di
        global di1

        if self.Model == "HKY":

            di=16
            di1=9
        else:
            di=3721
            di1=27

        if if_circle == True:
            self.making_Qg()

        if self.Q_new is None:
           self.making_Qg()

# Q_iiii is dialog
        Q_iiii = np.ones(di)
        for ii in range(di):
            Q_iiii[ii] = sum(self.Q[ii, ])

# max change number is 10
        max_number = 10

# dis_matrix can be igored
        time_list = []
        state_list = []
        dis_matrix = np.ones(self.sites_length)

# effect_matrix is used to store how many paths on each sites
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

            # time_list[ii] ==0 means there is a mutation, initial state not equal to  end state
            # Q_iiii means a diagonal entries of rate matrix


            if time_list[ii] == 0:

                time_matrix = 100*np.ones(shape=(repeat, max_number))
                state_matrix = 100*np.ones(shape=(repeat, max_number))

                for jj in range(repeat):
                        # most transfer 10 times
                        current_state = ini[ii]
                        i = 0
                        time = [100]
                        state = [0]
                        effect_number = 0
                        big_number = 0


                        while current_state != end[ii]:
                            current_state=ini[ii]
                            i = 1
                            time = [100]
                            state = [0]
                            u1 = np.random.uniform(0,1)
                            # Nelson approach
                            u= -np.log((1-(1-np.exp(-Q_iiii[int(current_state)]*t))*u1))/\
                                       (Q_iiii[int(current_state)])
                            time.append(u)
                            a = np.random.choice(range(di1), 1, p=self.Q_new[int(current_state), ])[0]
                            current_state = self.dic_col[int(current_state), a] - 1

                                    # if jump to absorbing state and without finishing process, we need to resample

                        #    while sum(self.Q_new[int(current_state), ]) == 0:
                           #         a = np.random.choice(range(di1), 1, p=self.Q_new[int(old), ])[0]
                           #         current_state = self.dic_col[int(old), a] - 1
                            state.append(int(current_state))


                            while u<=t:
                                    i=i+1
                                    u = u +  random.exponential(1/Q_iiii[int(current_state)])
                                    time.append(u)
                                    a = np.random.choice(range(di1), 1, p=self.Q_new[int(current_state), ])[0]
                                    current_state = self.dic_col[int(current_state), a] - 1
                                    # if jump to absorbing state and without finishing process, we need to resample
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
                        # store each path to matrix
                        else:
                            big_number = max(big_number, i)
                            if i > 0:
                                effect_number = (i-1) + effect_number
                            time_matrix[jj, 0:i] = time[0:i]
                            state_matrix[jj, 0:i] = state[0:i]
                            state_matrix[jj, 0] = ini[ii]
                        effect_matrix[ii,jj] = int(effect_number)
                        big_number_matrix[ii,jj] = int(big_number)


                time_list[ii]=time_matrix
                state_list[ii]=state_matrix

            # initial states is equal to end states
            else:

                time_matrix = 100 * np.ones(shape=(repeat, max_number))
                state_matrix = 100 * np.ones(shape=(repeat, max_number))

                for jj in range(repeat):

                    current_state = 11111
                    i=0

                    # most transfer 10 times
                    while current_state != end[ii]:
                          current_state = ini[ii]
                          i = 0
                          effect_number = 0
                          big_number = 0
                          u = 0
                          time = [100]
                          state = [100]
                          while u<=t:
                             u = u +  random.exponential(1/Q_iiii[int(current_state)])
                             if u<=t:
                                 i=i+1
                                 time.append(u)
                                 a = np.random.choice(range(di1), 1, p=self.Q_new[int(current_state),])[0]
                                 current_state = self.dic_col[int(current_state), a] - 1
                                # if jump to absorbing state and without finishing process, we need to resample
                                 state.append(int(current_state))



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
                            effect_number = i  + effect_number
                        time_matrix[jj, 0:(i+1)] = time[0:(i+1)]
                        state_matrix[jj, 0:(i+1)] = state[0:(i+1)]
                        state_matrix[jj, 0] = ini[ii]
                    effect_matrix[ii, jj] = int(effect_number)
                    big_number_matrix[ii, jj] = int(big_number)

                time_list[ii]=time_matrix
                state_list[ii]=state_matrix

        self.time_list=time_list
        self.state_list=state_list
        self.effect_matrix=effect_matrix
        self.big_number_matrix=big_number_matrix
        self.dis_matrix=dis_matrix




        return self.time_list,self.state_list,self.effect_matrix, self.big_number_matrix, self.dis_matrix

# ifrecall=True indicate we need do GLS_m again
# organize MC results from GLS_m method
    def GLS_s(self, t, repeat=1, ini=None, end=None, ifrecal=True,if_circle=False):

        if ifrecal==True:
            self.GLS_m(t=t, ini=ini, end=end, repeat = repeat,if_circle=if_circle)



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
                    big_number=int(max(big_number_matrix[i,a],big_number))
                    effect_number=effect_number+effect_matrix[i,a]
                    time_matrix_old = time_matrix
                    state_matrix_old = state_matrix
                    time_matrix = np.zeros(shape=(int(self.sites_length),int( big_number)))
                    state_matrix = np.zeros(shape=(self.sites_length, int(big_number)))
                    time_matrix[0:self.sites_length, 0:max_number] = time_matrix_old
                    state_matrix[0:self.sites_length, 0:max_number] = state_matrix_old
                    time_matrix[i, 0:big_number] = time_list[i][a,]
                    state_matrix[i, 0:big_number] = state_list[i][a,]

        time_matrix = time_matrix[0:self.sites_length, 0:int(big_number)]
        state_matrix = state_matrix[0:self.sites_length, 0:int(big_number)]
      #  print(time_matrix)
      #  print(state_matrix)



        return time_matrix, state_matrix, int(effect_number), int(big_number)


# method can be select as state or label
# the method state is more useful, since it can avoid the bias in sampling regarding small sample size
    def whether_IGC(self,history_matrix,effect_number,method="state",branch=2,times=0):

        p_h=np.zeros(shape=(effect_number, 9))


# 0 difference, 1 time, 2 whether igc, 3 paralog state,4 ratio time/tree length,5 branch,6times,

        if self.Model == "HKY":
            for ii in range(effect_number):
                p_h[ii, 7]=history_matrix[ii, 6]
                p_h[ii, 8] = history_matrix[ii, 7]
                p_h[ii, 5]=branch
                p_h[ii, 6] = times
                p_h[ii, 0]=history_matrix[ii,0]
                p_h[ii, 1] = history_matrix[ii, 4]
                p_h[ii, 4] = history_matrix[ii, 8]


                i_b=  int(history_matrix[ii, 6])//4
                j_b = int(history_matrix[ii, 6]) % 4
                i_p = int(history_matrix[ii, 7]) // 4
                j_p = int(history_matrix[ii, 7]) % 4

                if i_p == j_p:
                    if i_b != j_b and i_b == i_p:
                        # y_coor is corresponding coor for igc
                        y_coor = np.argwhere(self.dic_col[int(history_matrix[ii, 6]),] == (int(history_matrix[ii, 7]) + 1))[0]
                        qq=self.Q[int(history_matrix[ii, 6]), y_coor]

                        if method=="state":
                            p_h[ii, 2]=(self.tau)/qq

                        else:
                            u = random.uniform(0, 1)
                            if u<=float(self.tau)/qq:
                                p_h[ii, 2] =1

                    elif(i_b!=j_b and j_b==j_p):
                        y_coor=np.argwhere(self.dic_col[int(history_matrix[ii, 6]),] == (int(history_matrix[ii, 7]) + 1))[0]
                        qq=self.Q[int(history_matrix[ii, 6]),y_coor]
                        u=random.uniform(0,1)
                        if method=="state":
                            p_h[ii, 2]=(self.tau)/qq
                        else:
                            u = random.uniform(0, 1)
                            if u<=float(self.tau)/qq:
                                p_h[ii, 2] =1

        else:
            for ii in range(effect_number):
                p_h[ii, 7]=history_matrix[ii, 6]
                p_h[ii, 8] = history_matrix[ii, 7]
                p_h[ii, 5]=branch
                p_h[ii, 6] = times
                p_h[ii, 0]=history_matrix[ii,0]
                p_h[ii, 1] = history_matrix[ii, 4]
                p_h[ii, 4] = history_matrix[ii, 8]


                i_b=  int(history_matrix[ii, 6])//16
                j_b = int(history_matrix[ii, 6]) % 16
                i_p = int(history_matrix[ii, 7]) // 16
                j_p = int(history_matrix[ii, 7]) % 16

                if (i_p == j_p):
                    if (i_b != j_b and i_b == i_p):
                        y_coor = np.argwhere(self.dic_col[int(history_matrix[ii, 6]),] == (int(history_matrix[ii, 7]) + 1))[0]
                        qq = self.Q[int(history_matrix[ii, 6]), y_coor]
                        u = random.uniform(0, 1)

                        ca = geneconv.state_to_codon[j_b]
                        cb = geneconv.state_to_codon[j_p]

                        if isNonsynonymous(cb, ca, self.codon_table):
                            tau = self.tau * self.omega
                        else:
                            tau = self.tau

                        if method=="state":
                            p_h[ii, 2]=float(tau)/qq
                        else:
                            if u<=float(tau)/qq:
                                p_h[ii, 2] =1



                    elif (i_b != j_b and j_b==j_p):
                        y_coor = np.argwhere(self.dic_col[int(history_matrix[ii, 6]),] == (int(history_matrix[ii, 7]) + 1))[0]
                        qq = self.Q[int(history_matrix[ii, 6]), y_coor]
                        u = random.uniform(0, 1)

                        ca = geneconv.state_to_codon[i_b]
                        cb = geneconv.state_to_codon[i_p]

                        if isNonsynonymous(cb, ca, self.codon_table):
                            tau = self.tau * self.omega
                        else:
                            tau = self.tau

                        if method == "state":
                            p_h[ii, 2] = float(tau) / qq
                        else:
                            if u <= float(tau) / qq:
                                p_h[ii, 2] = 1

        for ii in range(effect_number-1):
            if p_h[ii,1]==0:
                p_h=p_h[0:ii-1,0:8]
                effect_number=ii-1
                break

       # print(p_h)


        return p_h, effect_number


# here time is a matrix
# rank the information from paths
    def rank_ts(self,t,time,state,ini,effect_number):



        if self.dic_di is None:
            self.making_dic_di()


        di=self.dic_di

        difference=0
        time_new=0

        global z
        z=False

        for i in range(self.sites_length):
            difference = difference+di[ini[i]]

    # 0 last difference number ,1 next difference number, 2 last time, 3 next time
    # 4 time difference is important, 5 location ,6 last state, 7 next state,8 ratio
        history_matrix = np.zeros(shape=(effect_number+1, 9))


        for jj in range(effect_number+1):

          #  print(time)
            coor = np.argmin(time)
            history_matrix[jj,0]=difference
            time_old=time_new
            history_matrix[jj, 2] = time_old
            time_new=np.min(time)
           # print(time_new)
            if(time_new>t):
                time_new=t
                z=True


            history_matrix[jj, 3] = time_new
            history_matrix[jj, 4] = time_new-time_old
            history_matrix[jj, 8]=history_matrix[jj, 4]/t
    # track state matrix
            d = time.shape[1]
            x_aixs = coor / d
            y_aixs = coor % d
            history_matrix[jj, 5]=x_aixs
            history_matrix[jj, 6] = ini[int(x_aixs)]
            history_matrix[jj, 7]=state[int(x_aixs), int(y_aixs)]

            history_matrix[jj, 1]=difference-di[int(history_matrix[jj, 6])]+di[int(history_matrix[jj, 7])]
            difference=history_matrix[jj, 1]
            # renew time matrix and ini stat
            time[int(x_aixs), int(y_aixs)]=100
            ini[int(x_aixs)]=history_matrix[jj, 7]
            if(z==True):
                effect_number=jj-1
                history_matrix[jj, 4]=0
                break;



        return history_matrix,effect_number


# organize information for all branches
    def monte_carlo(self,t=0.1,times=1,repeat=1,ifwholetree=True,ifpermutation=True,ifsave=False,
                   ):
        global re1
        global sitesd

        self.tauoriginal = deepcopy(self.tau)


        if ifpermutation==True:

            if ifwholetree == False:
                ini1 = self.make_ie(5, 8)[0]
                end1 = self.make_ie(5, 8)[1]
                re = self.GLS_s(t=t,repeat=repeat,ini=ini1,end=end1)

                sam = self.rank_ts(time=re[0],t=t, state=re[1], ini=ini1, effect_number=re[2])
                re1=self.whether_IGC(history_matrix=sam[0],effect_number=sam[1])
                effect_number=re1[1]
                re1=re1[0]

                for i in range(times-1):
                    re = self.GLS_s(t=t,repeat=repeat,ifrecal=False,ini=ini1,end=end1)
                    sam = self.rank_ts(time=re[0],t=t, state=re[1], ini=ini1, effect_number=re[2])
                    re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1])
                    re1=np.vstack((re1,re2[0]))
                    effect_number=effect_number+re2[1]


            else:
                ttt = len(self.scene['tree']["column_nodes"])
                for j in range(ttt):
                    t1 = self.scene['tree']["edge_rate_scaling_factors"][j]
                    print(j)
                    if j ==2:
                        ini2=self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                        end2 = self.geneconv.node_to_num[geneconv.edge_list[j][1]]

                        ini1 = self.make_ie(ini2, end2)[0]
                        end1 = self.make_ie(ini2, end2)[1]

                        re = self.GLS_s(t=t1,repeat=repeat,ini=ini1,end=end1)
                        sam = self.rank_ts(time=re[0], t=t1,state=re[1], ini=ini1, effect_number=re[2])
                        re1=self.whether_IGC(history_matrix=sam[0],effect_number=sam[1])
                        max_diff=0
                        di = self.dic_di
                        for i in range(self.sites_length):
                            max_diff = max_diff + di[end1[i]]
                        self.min_diff=max_diff
                        effect_number=re1[1]
                        re1=re1[0]


                        for i in range(times-1):
                            re = self.GLS_s(t=t1,repeat=repeat,ifrecal=False,ini=ini1,end=end1)
                            sam = self.rank_ts(time=re[0],t=t1, state=re[1], ini=ini1, effect_number=re[2])
                            re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1])
                            re1=np.vstack((re1,re2[0]))
                            effect_number=effect_number+re2[1]




                    elif j==1:
                        print("ignore the outgroup")


                    elif  j>2:
                        ini2=self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                        end2 = self.geneconv.node_to_num[geneconv.edge_list[j][1]]
                        #print(ini2)
                        #print(end2)
                        ini1 = self.make_ie(ini2, end2)[0]
                        end1 = self.make_ie(ini2, end2)[1]

                        re = self.GLS_s(t=t1,repeat=repeat,ini=ini1,end=end1)

                        sam = self.rank_ts(time=re[0], t=t1,state=re[1], ini=ini1, effect_number=re[2])
                        re2=self.whether_IGC(history_matrix=sam[0],effect_number=sam[1])
                        effect_number1=re2[1]
                        re2=re2[0]

                        for i in range(times-1):
                            re = self.GLS_s(t=t1,ifrecal=False,repeat=repeat,ini=ini1,end=end1)
                            sam = self.rank_ts(time=re[0], t=t1,state=re[1], ini=ini1, effect_number=re[2])
                            re3 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1])
                            re2=np.vstack((re2,re3[0]))
                            effect_number1=effect_number1+re3[1]


                        re1 = np.vstack((re1, re2))
                        effect_number=effect_number1+effect_number

        else:

            if ifwholetree == False:

                ini1 = self.make_ie(0, 1)[0]
                end1 = self.make_ie(0, 1)[1]

                re = self.GLS_s(t=t, repeat=1, ini=ini1, end=end1)


                sam = self.rank_ts(time=re[0], t=t,state=re[1], ini=ini1, effect_number=re[2])
                re1 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1])
                effect_number = re1[1]
                re1 = re1[0]


                for i in range(times - 1):
                    re = self.GLS_s(t=t, repeat=1, ifrecal=True, ini=ini1, end=end1)
                    sam = self.rank_ts(time=re[0],t=t, state=re[1], ini=ini1, effect_number=re[2])
                    re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1])
                    re1 = np.vstack((re1, re2[0]))
                    effect_number = effect_number + re2[1]


            else:
                kk=0
                effect_number = 0
                name=False
                ttt = len(self.scene['tree']["column_nodes"])
                while(kk<times):
                    kk=kk+1
                    self.jointly_common_ancstral_inference()
                    sitesd=deepcopy(self.sites)
                    print(kk)
                    for j in range(ttt):
                             t1 = self.scene['tree']["edge_rate_scaling_factors"][j]
                             if not j == 1:
                                  ini2 = self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                                  end2 = self.geneconv.node_to_num[geneconv.edge_list[j][1]]
                                  ini1=deepcopy(self.sites[ini2,])
                                  end1=deepcopy(self.sites[end2,])
                                #  self.measure_difference(ini1,end1,j)

                                  re = self.GLS_s(t=t1, repeat=1, ifrecal=True,ini=ini1, end=end1)

                                  if re[2]==0:
                                      break



                                  sam = self.rank_ts(time=re[0],t=t1, state=re[1], ini=ini1, effect_number=re[2])
                                  re10 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1],branch=j,times=kk)
                         #         print(re10[1])
                                  if(name==False):
                                      re111=re10[0]
                                      name=True
                                  else:
                                      re111=np.vstack((re111, re10[0]))

                                  effect_number = effect_number+re10[1]

                            #      re111.append(re10[0])

                    else:
                        continue


  #      if ifsave==True:

     #      save_nameP = '../test/savesample/Ind_' + geneconv.Model+ geneconv.paralog[0]+geneconv.paralog[1]+'sample.txt'
     #      np.savetxt(open(save_nameP, 'w+'), re111.T)



        return re111 , effect_number


# this is using to test the estimation of tau given real history

    def monte_carlo_s(self,ifsave=True,
                    iftestsimulation=True,sizen=9999,times=1):

        self.change_t_Q(tau=1)
        self.tau=0.8
        self.sites_length=sizen
        list=[]

        if iftestsimulation==True:

            ttt = len(self.scene['tree']["column_nodes"])
            for j in range(ttt):
                t1 = 0.2
                print(j)

                if j == 0:
                    ini1 = self.make_ini(sizen=sizen)

                    re=self.GLS_si(t=t1,ini=ini1,sizen=sizen)
                    list.append(re[1])
                    re1=None
                    effect_number=0

                elif j == 1:
                    print("ignore the outgroup")


                elif j >= 2:

                    ini2 = self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                    re=self.GLS_si(t=t1,ini=list[int(ini2//2)],sizen=sizen)
                    if j%2==0:
                        list.append(re[1])

                    sam = self.rank_ts(time=re[2], t=t1, state=re[3], ini=re[0], effect_number=re[4])
                    re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1],branch=j)
                    effect_number1 = re2[1]
                    re2 = re2[0]

                    di=0
                    for jj in range(sizen):
                            di=self.dic_di[re[1][jj]]+di

                    print(di)

                    if re1 is  None:
                        re1=re2

                    re1 = np.vstack((re1, re2))
                    effect_number = effect_number1 + effect_number



        if ifsave==True:

           save_nameP = '../test/savesample/Ind_' + geneconv.Model+ geneconv.paralog[0]+geneconv.paralog[1]+'simulation.txt'
           np.savetxt(open(save_nameP, 'w+'), re1.T)

        return re1 , effect_number

# this  is using to test the estimation of tau given real internal node
    def monte_carlo_s1(self,ifsave=True,times=1,
                    iftestsimulation=True,sizen=9999):

        self.change_t_Q(tau=0.4)
        self.tau=0.4
        self.sites_length=sizen
        aaa = self.topo1(sizen=sizen)


        if iftestsimulation==True:

            for kk in range(times):
                print(kk)
                ttt = len(self.scene['tree']["column_nodes"])
                if kk == 0:
                    for j in range(ttt):
                        t1 = 0.2

                        if j == 2:
                            ini2 = self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                            end2 = self.geneconv.node_to_num[geneconv.edge_list[j][1]]
                            ini1 = aaa[ini2]
                            end1 = aaa[end2]

                            re = self.GLS_s(t=t1, repeat=1, ifrecal=True, ini=ini1, end=end1)
                            sam = self.rank_ts(time=re[0], t=t1, state=re[1], ini=ini1, effect_number=re[2])
                            re10 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1],branch=j)
                            effect_number = re10[1]
                            re1 = re10[0]

                            di = 0
                            for jj in range(sizen):
                                di = self.dic_di[end1[jj]] + di

                            print(di)

                        elif j == 1:
                            print("ignore the outgroup")


                        elif j > 2:
                            ini2 = self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                            end2 = self.geneconv.node_to_num[geneconv.edge_list[j][1]]
                            ini1 = aaa[ini2]
                            end1 = aaa[end2]

                            re = self.GLS_s(t=t1, ifrecal=True, repeat=1, ini=ini1, end=end1)

                            sam = self.rank_ts(time=re[0], t=t1, state=re[1], ini=ini1, effect_number=re[2])
                            # print(sam[0])
                            re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1],branch=j)
                            effect_number1 = re2[1]
                            re2 = re2[0]

                            re1 = np.vstack((re1, re2))
                            effect_number = effect_number1 + effect_number


                else:
                    for j in range(ttt):
                        t1 = self.scene['tree']["edge_rate_scaling_factors"][j]
                        # print(j)

                        if j == 2:
                            ini2 = self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                            end2 = self.geneconv.node_to_num[geneconv.edge_list[j][1]]
                            ini1 = aaa[ini2]
                            end1 = aaa[end2]

                            re = self.GLS_s(t=t1, repeat=1, ifrecal=True, ini=ini1, end=end1)
                            sam = self.rank_ts(time=re[0], t=t1, state=re[1], ini=ini1, effect_number=re[2])
                            re10 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1],branch=j)
                            max_diff = 0
                            di = self.dic_di
                            for i in range(self.sites_length):
                                max_diff = max_diff + di[end1[i]]
                            self.min_diff = max_diff
                            effect_numbern = re10[1]
                            re1n = re10[0]

                        elif j == 1:
                            print("ignore the outgroup")


                        elif j > 2:
                            ini2 = self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                            end2 = self.geneconv.node_to_num[geneconv.edge_list[j][1]]
                            ini1 = aaa[ini2]
                            end1 = aaa[end2]

                            re = self.GLS_s(t=t1, ifrecal=True, repeat=1, ini=ini1, end=end1)

                            sam = self.rank_ts(time=re[0], t=t1, state=re[1], ini=ini1, effect_number=re[2])
                            # print(sam[0])
                            re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1],branch=j)
                            effect_number1n = re2[1]
                            re2n = re2[0]

                            re1n = np.vstack((re1n, re2n))
                            effect_numbern = effect_number1n + effect_numbern

                    re1 = np.vstack((re1, re1n))
                    effect_number = effect_number + effect_numbern


        return re1 , effect_number

### divide the paralog diverge number
    def divide_Q(self, times, repeat,method="simple", simple_state_number=5,ifpermutation=True,ifsave=True,
                 ifsimulation=False):

        if ifsimulation==True:
            re=self.monte_carlo_s_EM(ifsave=ifsave,times=times)

        else:
            re=self.monte_carlo(times=times,repeat=repeat,ifpermutation=ifpermutation,ifsave=ifsave)

        history_matrix=re[0]
        effect_number=re[1]
        type_number = simple_state_number
        self.type_number=int(type_number)
        self.last_effct=int(effect_number)


        if (method == "simple"):
            quan = 1 / float(simple_state_number)
            quan_c = quan
            stat_rank = pd.DataFrame(history_matrix[:, 0])
            stat_vec = np.zeros(shape=(simple_state_number - 1, 1))
            for i in range(simple_state_number - 1):
                stat_vec[i] = np.quantile(stat_rank, quan_c)
                quan_c = quan_c + quan

            print(stat_vec)
            for ii in range(effect_number):
                if (history_matrix[ii, 0] <= stat_vec[0]):
                    history_matrix[ii, 3] = 0
                elif (history_matrix[ii, 0] > stat_vec[simple_state_number - 2]):
                    history_matrix[ii, 3] = simple_state_number - 1
                else:
                    for iii in range(simple_state_number - 1):
                        if (history_matrix[ii, 0] <= stat_vec[iii + 1] and history_matrix[ii, 0] > stat_vec[iii]):
                            history_matrix[ii, 3] = iii + 1
                            break


        elif (method=="divide"):
            zzz = np.argmax(history_matrix[:, 0])
         #   big_quan = history_matrix[zzz, 0] / self.sites_length
            max_quan = history_matrix[zzz, 0]
            print(max_quan)
            zzz = np.argmin(history_matrix[:, 0])
            #   big_quan = history_matrix[zzz, 0] / self.sites_length
            min_quan = history_matrix[zzz, 0]
            print(min_quan)
            #stat_rank = pd.DataFrame(history_matrix[:, 0])
            #min_quan=np.quantile(stat_rank, 0.05)

            quan=(max_quan-min_quan)/(simple_state_number-1)
            quan_c = quan+min_quan

            stat_vec = np.zeros(shape=(simple_state_number - 1, 1))
            for i in range(simple_state_number - 1):
                stat_vec[i] = quan_c
                quan_c = quan_c + quan

            print(stat_vec)
            for ii in range(effect_number):
                if (history_matrix[ii, 0] <= stat_vec[0]):
                    history_matrix[ii, 3] = 0
                elif (history_matrix[ii, 0] > stat_vec[simple_state_number - 2]):
                    history_matrix[ii, 3] = simple_state_number - 1
                else:
                    for iii in range(simple_state_number - 2):
                        if (history_matrix[ii, 0] <= stat_vec[iii + 1] and history_matrix[ii, 0] > stat_vec[iii]):
                            history_matrix[ii, 3] = iii + 1
                            break

##########################
#########branch number-1
        elif(method=="bybranch"):
            type_number = len(self.scene['tree']["column_nodes"])-1

            self.type_number = int(type_number)
            for ii in range(effect_number):
                    history_matrix[ii, 3] = history_matrix[ii, 5]-2


        self.igc_com=deepcopy(history_matrix)
      #  print(history_matrix)

        return history_matrix

# tolenrene
    def EM_change_tau(self,max_time=50,tolenrence=0.0001,times=2,method="divide",
                      repeat=1,simple_state_number=10,ifpermutation=True,ifsave=True,
                      ifcircle=False):

        if ifcircle==False:
             self.divide_Q(times=times, repeat=repeat, simple_state_number=simple_state_number,
                      ifpermutation=ifpermutation, ifsave=ifsave, method=method)

        ifcorrect=False



        list_all=[]
        for i in range(self.type_number - 1):
            list_all.append(0)

        effect_vector=np.zeros(shape=int(self.type_number - 1))
        for j in range(self.last_effct):
            for i in range(self.type_number - 1):
                if(self.igc_com[j, 3] == i):
                    effect_vector[i]=int(effect_vector[i])+1

      #  reorder the history


        for i in range(self.type_number - 1):
            kk=0
            list = np.ones(shape=(int(effect_vector[i]), 9))
            for j in range(self.last_effct):
                if(self.igc_com[j, 3] == i):
                    list[kk,]=(self.igc_com[j, ])
                    kk=kk+1

            list_all[i]=deepcopy(list)

        print(effect_vector)
        tau_list=np.ones(shape=int(self.type_number - 1))

        for i in range(self.type_number - 1):
            ii=0
            eps=1000
          #  print(i)
            while(ii<=max_time and eps>=tolenrence):
                tau = np.zeros(shape=(times))
                deno1 = np.zeros(shape=(times))
                total_history=0
                total_IGC=0
                times1=times
          #      print("xxxxxxxxxxxxxxxxxxxx")
                for k in range(times):
                    deno = 0
                    no = 0
                    for j in range(int(effect_vector[i])):
                        if (int(list_all[i][j,6]) == k):
                                deno = deno + (list_all[i][j,1] * list_all[i][j,0])
                                no = no + list_all[i][j,2]
                                total_IGC=total_IGC+list_all[i][j,2]
                                total_history = total_history +1

                #    print(deno)
                    if deno==0:
                        tau[k]=0
                        times1=times1-1
                    else:
                        tau[k] = ((no)/(deno * 2))
                    deno1[k]=deno

                if times1==0:
                    times1=1

                if ifcorrect==False:
                    tau_list[i]= deepcopy(np.sum(tau)/times1)
                else:
                    ps_bybranch=self.sites_length*2*(self.scene['tree']["edge_rate_scaling_factors"][i+2])
                    if((total_history/times)>ps_bybranch):
                        tau_list[i]=0.9*np.mean(tau)+0.1*(((total_history/times)-ps_bybranch)/np.mean(deno1))*0.5
                    else:
                        tau_list[i] = deepcopy(np.mean(tau))

            #    if tau_list[i]==0:
            #      tau_list[i]=deepcopy(self.tauoriginal)


                eps=np.abs(tau_list[i]-deepcopy(self.tau))
                self.change_t_Q(tau=tau_list[i])
                list_all[i]=deepcopy(self.whether_IGC_EM(history_matrix=list_all[i], effect_number=int(effect_vector[i])))
                ii=ii+1


            print("estimated  events:",(total_history/times1))
            print("estimated  IGC", (total_IGC / times1))
            print("estimated  ps", ((total_history/times1)-(total_IGC / times1)))
            print("estimaed branch len",(self.scene['tree']["edge_rate_scaling_factors"][i+2]))
            print("IGC",tau_list[i])

        return list_all, tau_list

    def MC_EM(self,max_time=50,tolenrence=0.0001,times=2,method="divide",
                      repeat=1,simple_state_number=10,ifpermutation=True,ifsave=True,
                  EM_circle=3,ifcircle=True):

        fr=self.EM_change_tau(times=times, max_time=max_time,tolenrence=tolenrence,repeat=repeat, ifpermutation=ifpermutation, method="bybranch")

        print(fr[1])

      #  fr1=np.array([1,1,1,1,0.1,0.1],dtype=float)

        for circle in range(EM_circle):
            kk=0
            effect_number=0
            name = False
            ttt = len(self.scene['tree']["column_nodes"])
            while (kk <times):
                kk=kk+1
                self.change_t_Q(tau=self.tauoriginal)
                if ifcircle==True:
                     self.jointly_common_ancstral_inference(ifcircle=ifcircle,taulist=fr[1])
                #      self.jointly_common_ancstral_inference(ifcircle=ifcircle, taulist=fr1)
                print(kk)
                for j in range(ttt):
                        t1 = self.scene['tree']["edge_rate_scaling_factors"][j]
                        # print(j)

                        if not j == 1:
                            ini2 = self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                            end2 = self.geneconv.node_to_num[geneconv.edge_list[j][1]]
                            ini1 = deepcopy(self.sites[ini2,])
                            end1 = deepcopy(self.sites[end2,])
                            self.change_t_Q(tau=fr[1][j-2])
                          #  self.change_t_Q(tau=fr1[j - 2])

                            re = self.GLS_s(t=t1, repeat=1, ifrecal=True, ini=ini1, end=end1,if_circle=True)
                            if re[2] == 0:
                                break

                            sam = self.rank_ts(time=re[0], t=t1, state=re[1], ini=ini1, effect_number=re[2])
                            re10 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1], branch=j, times=kk)

                            if (name == False):
                                re111 = re10[0]
                                name = True
                            else:
                                re111 = np.vstack((re111, re10[0]))
                            effect_number = effect_number+re10[1]
                else:
                        continue


            type_number = len(self.scene['tree']["column_nodes"]) - 1
            self.type_number = int(type_number)
            self.last_effct = int(effect_number)
            for ii in range(effect_number):
                re111[ii, 3] = re111[ii, 5] - 2
            self.igc_com = deepcopy(re111)
            fr = self.EM_change_tau(times=times, repeat=repeat, ifpermutation=ifpermutation,
                                    method="bybranch",ifcircle=True)
            print(fr[1])

        history_matrix=deepcopy(self.igc_com)
        type_number = deepcopy(simple_state_number)
        self.type_number = int(type_number)



        if (method == "simple"):
                quan = 1 / float(simple_state_number)
                quan_c = quan
                stat_rank = pd.DataFrame(history_matrix[:, 0])
                stat_vec = np.zeros(shape=(simple_state_number - 1, 1))
                for i in range(simple_state_number - 1):
                    stat_vec[i] = np.quantile(stat_rank, quan_c)
                    quan_c = quan_c + quan

                print(stat_vec)
                for ii in range(effect_number):
                    if (history_matrix[ii, 0] <= stat_vec[0]):
                        history_matrix[ii, 3] = 0
                    elif (history_matrix[ii, 0] > stat_vec[simple_state_number - 2]):
                        history_matrix[ii, 3] = simple_state_number - 1
                    else:
                        for iii in range(simple_state_number - 1):
                            if (history_matrix[ii, 0] <= stat_vec[iii + 1] and history_matrix[ii, 0] > stat_vec[iii]):
                                history_matrix[ii, 3] = iii + 1
                                break


        self.igc_com=deepcopy(history_matrix)
        print(self.igc_com[1,])




### get estimation
    def get_igcr_pad(self,times=2, repeat=1,max_time=50,tolenrence=0.0001,simple_state_number=10,ifpermutation=False,ifsave=True,
                     method="divide",EM_circle=5):

             self.MC_EM(times=times,repeat=repeat,max_time=max_time,tolenrence=tolenrence,simple_state_number=simple_state_number,
                           ifpermutation=ifpermutation,ifsave=ifsave,method=method,EM_circle=EM_circle)
             ## self.igc_com 0 difference number between paralog, 1 occupancy time for interval ,2 igc_number,3 state,4 propption = occupancy time/branch length

             relationship=np.zeros(shape=(self.type_number-1, 8))


             ## relation ship 0 denominator of igc : time * difference,
             ## 1 sum of igc events, 2 total time (may be usless),3 ratio: igc rates
             for i in range(self.type_number-1):
                tau=np.zeros(shape=(times))
                times1=times
                total_igc = 0
                total_history = 0

                for k in range(times):
                    deno = 0
                    no = 0
                    total_time = 0
                    total_pro = 0

                    for j in range(self.last_effct):
                        if(self.igc_com[j,6]==k):
                           if(self.igc_com[j,3]==i):
                               total_history=total_history+1
                               deno = deno + (self.igc_com[j, 1] * self.igc_com[j, 0])
                               no=no+self.igc_com[j, 2]
                               total_time=total_time+self.igc_com[j, 1]
                               total_pro=total_pro+self.igc_com[j, 4]
                               if(self.igc_com[j,2]>0):
                                  total_igc=total_igc+1

                    if (deno==0):
                        tau[k]=0
                        times1=times1-1
                    else:
                        tau[k]=((no)/(deno*2))

                if times1==0:
                     times1=1


                relationship[i,0]=deno
                relationship[i, 1] =no
                relationship[i, 2] = total_time
                relationship[i, 3] = np.sum(tau)/times1
                tauratio=deepcopy(tau)
                tausquare=deepcopy(tau)
                for k in range(times):
                    tauratio[k]=(relationship[i,3])/(2*tauratio[k])
                    tausquare[k]=(tausquare[k]**2)
                relationship[i, 4] = np.mean(tausquare) + np.mean(tauratio)-(relationship[i, 3]**2)
                relationship[i, 5]=total_igc/times1
                relationship[i, 6] = total_history/times1
                relationship[i, 7] = (total_history-no)/times1


                if(method=="bybranch"):
                     print("estimated branch length:", self.scene['tree']["edge_rate_scaling_factors"][i + 2])
                     print("estimated branch name:", self.geneconv.edge_list[i + 2])
                     print("for branch :" , geneconv.edge_list[i+2])
                     print("total events :" , (total_history/times1))
                     print("total igc :" , (no/times1))
                # point mutation = events- igc
                     print("total point mutations :", relationship[i, 7])
                     print("estimated tau :", relationship[i, 3])
                     print("estimated var of tau :", relationship[i, 4])
                     print(np.mean(tauratio))
                     print("#######################")

                else:
                     print("total events :" , relationship[i,6])
                     print("total potential igc :" , relationship[i,5])
                # point mutation = events- igc
                     print("total point mutations :"  ,relationship[i, 7])
                     print("estimated tau :" , relationship[i, 3])
                     print("estimated var of tau :", relationship[i, 4])
                     print("##########")


             save_nameP = '../test/savesample/Ind_re_' + geneconv.Model + geneconv.paralog[0] + geneconv.paralog[
                 1] + 'sample.txt'
             with open(save_nameP, 'w+') as f:
                 np.savetxt(f, relationship.T)

             self.relationship=relationship


             return relationship

# we can select different kernel method , the default one is "exp"

    def get_parameter(self,function="linear"):

        if function=="exp":
 #           igc=np.sum(self.igc_com[:,2])
            pro=0
            for i in range(self.last_effct):
                pro=(1-(self.igc_com[i,0]/self.sites_length))*(self.igc_com[i,1])+pro

            alpha=pro/self.last_effct

        if function == "linear":
                igc = np.sum(self.igc_com[:, 2])
                pro = 0
                for i in range(self.last_effct):
                    pro = (self.igc_com[i, 0]) * (self.igc_com[i, 1]) + pro

                alpha = igc / pro

        if function == "squre":
                igc = np.sum(self.igc_com[:, 2])
                pro = 0
                for i in range(self.last_effct):
                    pro = (self.igc_com[i, 0] ** 2)* (self.igc_com[i, 1]) + pro

                alpha = igc / pro


        return alpha

# here is a function to store some data which may be easily apploed
    def store_vector(self):
        # data for EDN ECP
        branch_length=[0.071054170, 0.103729546, 0.008834915 ,0.051509262 ,0.010990042 ,0.030066857 ,0.004586267,
                       0.005039781]
        tau=[1.805921]



##################################
# This part is used for simulation
# sizen/3 %==0
#################################
    def make_ini(self,sizen):
            ini = np.ones(sizen)
            z = self.geneconv.pi


            if self.Model=="HKY":
                sample=np.ones(4)
                for i in range(16):
                    if(i // 4 == i%4):
                        sample[i%4]=i

                for i in range(sizen):
                    ini[i] = int(np.random.choice(sample, 1, p=(z))[0])

            else:
                sample = np.ones(61)
                for i in range(3721):
                    if (i // 61 == i % 61):
                        sample[i % 61] = i

                for i in range(sizen):
                    ini[i] = int(np.random.choice(sample, 1, p=(1 / float(61)) * np.ones(61))[0])

            return (ini)

    def GLS_si(self,t=0.02,ini =None,sizen=150,tau=0.1,ifdet=True):

      #  if self.Q_new is None:
        self.making_Qg()

        global di
        global di1

        if self.Model == "HKY":
            di=16
            di1=9

        else:
            di=3721
            di1=27

        Q_iiii = np.ones((di))
        for ii in range(di):
            Q_iiii[ii] = sum(self.Q[ii,])

        end = np.ones(sizen)

        time_matrix = 100 * np.ones(shape=(sizen, 10))
        state_matrix = np.zeros(shape=(sizen, 10))
        effect_number=0


        if ifdet==False:

            for ll in range(sizen):

                    curent_state = ini[ll]
                    u = random.exponential(1/Q_iiii[int(curent_state)])
                    i=0

                    while(u<=t):
                        i=i+1
                        a = np.random.choice(range(di1), 1, p=self.Q_new[int(curent_state),])[0]
                        curent_state = self.dic_col[int(curent_state), a] - 1
                        time_matrix[ll,i]=u
                        state_matrix[ll,i]=int(curent_state)
                        u=u+random.exponential(1/Q_iiii[int(curent_state)])


                    effect_number=i+effect_number
                    end[ll]=int(curent_state)

# simulate for testing
        if ifdet==True:

            for ll in range(sizen):
                  curent_state = ini[ll]
                  u = random.exponential(1 / Q_iiii[int(curent_state)])
                  i = 0

                  while (u <= t):
                         i = i + 1
                         a = np.random.choice(range(di1), 1, p=self.Q_new[int(curent_state),])[0]
                         curent_state = self.dic_col[int(curent_state), a] - 1

                         time_matrix[ll, i] = u
                         state_matrix[ll, i] = int(curent_state)
                         u = u + random.exponential(1 / Q_iiii[int(curent_state)])

                  effect_number = i + effect_number
                  end[ll] = int(curent_state)


            sam = deepcopy(self.rank_ts(time=time_matrix, t=t, state=state_matrix, ini=deepcopy(ini), effect_number=effect_number))
      #      self.change_t_Q(tau)
            re2 = deepcopy(self.whether_IGC(history_matrix=sam[0], effect_number=sam[1], branch=1, times=0))
            history=re2[0]

            total_history=0
            deno=0
            no=0

            for i in range(re2[1]):

               total_history = total_history + 1
               deno = deno + (history[i, 1] * history[i, 0])
               no = no + history[i, 2]


            print("total events :", total_history )
            print("total igc :", no)
            print("total point mutation :", total_history-no)
            print("IGC rate",no/(deno*2))



        return ini,end,time_matrix,state_matrix,effect_number,10


    def remake_matrix(self):
        if self.Model=="HKY":
            Q = geneconv.get_HKYBasic()
            print(Q)

        if self.Model=="MG94":
            Q=geneconv.get_MG94Basic()

        return Q


# used  before topo so  that can make new Q
    def change_t_Q(self,tau=99):

        if self.Q is None:
           self.making_Qmatrix()


        if self.Model == "HKY":
            for ii in range(16):
                for jj in range(9):
                    i_b=ii//4
                    j_b=ii%4
                    i_p=(self.dic_col[ii,jj]-1)//4
                    j_p=(self.dic_col[ii,jj]-1)%4
                    if i_p == j_p:
                        if i_b != j_b and i_b == i_p:
                            self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau
                        elif (i_b != j_b and j_b == j_p):
                            self.Q[ii, jj] = self.Q[ii, jj] - self.tau + tau

        else:

            for ii in range(61):
                for jj in range(27):
                    if ii==self.dic_col[ii,jj]-1:
                        self.Q[ii,jj]=self.Q[ii,jj]-self.tau+tau

        self.tau=tau

        # used  before topo so  that can make new Q


### this one is more flexiable

    def trans_into_seq(self,ini=None,leafnode=4,sizen=0):
        list = []

        if self.Model == 'MG94':
            dict = self.geneconv.state_to_codon
            for i in range(leafnode):
                p0 = ">paralog0"
                p1 = ">paralog1"
                for j in range(sizen):
                    p0 = p0 + dict[(ini[i][j]) // 61]
                    p1 = p1 + dict[(ini[i][j]) % 61]
                list.append(p0)

                list.append(p1)
        else:
            dict = self.geneconv.state_to_nt
            for i in range(leafnode):
                p0 = "\n"+">paralog0"+"\n"
                p1 = "\n"+">paralog1"+"\n"
                for j in range(sizen):
                    p0 = p0 + dict[(ini[i][j]) // 4]
                    p1 = p1 + dict[(ini[i][j]) % 4]


                list.append(p0)
                list.append(p1)

            p0 = "\n"+">paralog0"+"\n"
            for j in range(sizen):
                p0 = p0 + dict[(ini[leafnode][j])]

            list.append(p0)

        save_nameP = '../test/savesample/' + 'sample1.txt'
        with open(save_nameP, 'wb') as f:
            pickle.dump(list, f)


        return (list)



### calculat how different of paralog:
    def difference(self,ini,selecr=(3,4),sizen=999):


        Q=self.remake_matrix()
        for  i in range(4):
            Q[i,i]=sum(-Q[i,])
        Q=linalg.expm(Q*1.2)



        if self.Model == 'MG94':
            dict = self.geneconv.state_to_nt
            site= np.zeros(shape=(61, 61))
            for j in  range(sizen):
                    p0 =  dict[(ini[selecr[0]][j]) // 61]
                    p1 =  (ini[selecr[1]][j])
                    site[p0][p1]=site[p0][p1]+1


        else:
            site= np.zeros(shape=(4, 4))
            site1 = np.zeros(shape=(4, 4))

            for j in  range(sizen):
                    p0 =  int((ini[selecr[0]][j]) // 4)
                    p1 =  int((ini[selecr[1]][j]))
                    site[p0,p1]=int(site[p0,p1])+1
                    site1[p0,]=Q[p0,]+site1[p0,]

            #print(Q)
            for i in range(4):
                Q[i,] =self.geneconv.pi*Q[i,]*sizen
               # print(sum(site1[i,]))

            #print(Q)



    ##### topology is pretty simple

    def topo(self,leafnode=4,sizen=999,t=0.1):
        self.sites_length=sizen
        ini=self.make_ini(sizen=sizen)
###### calculate the out group

        list=[]
        if self.Model=="HKY":

            Q = self.remake_matrix()
            end1 = np.ones(sizen)
            Q_iiii = np.ones((4))
            for ii in range(4):
                qii = sum(Q[ii,])
                if qii != 0:
                    Q_iiii[ii] = sum(Q[ii,])

            for d in range(4):
                Q[d,] = Q[d,] / Q_iiii[d]



            for ll in range(sizen):
                # most transfer 5 time
                    curent_state = ini[ll]//4
                    u = random.exponential(1/Q_iiii[int(curent_state)])
                    while(u<=t):
                        a = np.random.choice(range(4), 1, p=Q[int(curent_state),])[0]
                        curent_state = a
                        u=u+random.exponential(1/Q_iiii[int(curent_state)])

                    end1[ll]=curent_state

        # append ini
            list1=[]
            list1.append(ini)
            mm=np.ones(shape=(4, sizen))
            mm[0,:]=ini



        for i in range(leafnode):

            if(i== leafnode-1):
                leaf = self.GLS_si(ini=deepcopy(ini), sizen=sizen,ifdet=True)[1]
                list.append(leaf)

            elif(i== leafnode-2):
                # ini is internal node, leaf is observed;
                # list store observed
                ini = deepcopy(self.GLS_si(ini=deepcopy(ini), sizen=sizen)[1])
               # self.change_t_Q(0.1)
                leaf = deepcopy(self.GLS_si(ini=deepcopy(ini), sizen=sizen,ifdet=True)[1])
                list.append(leaf)
                list1.append(ini)
                mm[i+1, :] = ini

            else:
                # ini is internal node, leaf is observed;
                # list store observed
                ini = deepcopy(self.GLS_si(ini=deepcopy(ini), sizen=sizen)[1])
                leaf = deepcopy(self.GLS_si(ini=deepcopy(ini), sizen=sizen,ifdet=True,tau=1)[1])
                list.append(leaf)
                list1.append(ini)
                mm[i+1, :] = ini
            self.measure_difference(ini,leaf,1)

        # append outgroup
        list.append(end1)

        save_nameP = '../test/savesample/RRR_Internal_' +  geneconv.paralog[0] + geneconv.paralog[
            1] + 'sample.txt'


        mm=np.array(mm,order="F")
        np.savetxt(save_nameP,mm)

      #  with open(save_nameP, 'wb') as f:
        #    pickle.dump(mm, f)


        return list

    ##### topology contain internal node used to do test
    def topo1(self,leafnode=4,sizen=999,t=0.4):
        ini=self.make_ini(sizen=sizen)

###### calculate the out group

        list=[]
        if self.Model=="HKY":

            Q = self.remake_matrix()
            end1 = np.ones(sizen)
            Q_iiii = np.ones((4))
            for ii in range(4):
                qii = sum(Q[ii,])
                if qii != 0:
                    Q_iiii[ii] = sum(Q[ii,])

            for d in range(4):
                Q[d,] = Q[d,] / Q_iiii[d]



            for ll in range(sizen):
                # most transfer 5 time
                    curent_state = ini[ll]//4
                    u = random.exponential(1/Q_iiii[int(curent_state)])
                    while(u<=t):
                        a = np.random.choice(range(4), 1, p=Q[int(curent_state),])[0]
                        curent_state = a
                        u = random.exponential(1 / Q_iiii[int(curent_state)])

                    end1[ll]=curent_state


            list.append(ini)
            mm=np.ones(shape=(4, sizen))
            mm[0,:]=ini

        for i in range(leafnode):

            if(i== leafnode-1):
                leaf = self.GLS_si(ini=ini, sizen=sizen)[1]
                list.append(leaf)


            elif(i==0):
                # ini is internal node
                ini = self.GLS_si(ini=ini, sizen=sizen)[1]
                leaf = self.GLS_si(ini=ini, sizen=sizen)[1]
                list.append(ini)
                list.append(end1)
                list.append(leaf)
                mm[i+1, :] = ini

            else:
                ini = self.GLS_si(ini=ini, sizen=sizen)[1]
                leaf = self.GLS_si(ini=ini, sizen=sizen)[1]
                list.append(ini)
                list.append(leaf)
                mm[i + 1, :] = ini

        return list



    def topo_EDNECP(self, sizen=999):
            self.sites_length = sizen
            ini = self.make_ini(sizen=sizen)
            branch_length = [0.071054170, 0.103729546, 0.008834915, 0.051509262, 0.010990042, 0.030066857, 0.004586267,
                             0.005039781]

            list = []
            if self.Model == "HKY":

                Q = self.remake_matrix()
                end1 = np.ones(sizen)
                Q_iiii = np.ones((4))
                for ii in range(4):
                    qii = sum(Q[ii,])
                    if qii != 0:
                        Q_iiii[ii] = sum(Q[ii,])

                for d in range(4):
                    Q[d,] = Q[d,] / Q_iiii[d]

                for ll in range(sizen):
                    # most transfer 5 time
                    curent_state = ini[ll] // 4
                    u = random.exponential(1 / Q_iiii[int(curent_state)])
                    while (u <= branch_length[1]):
                        a = np.random.choice(range(4), 1, p=Q[int(curent_state),])[0]
                        curent_state = a
                        u = u + random.exponential(1 / Q_iiii[int(curent_state)])

                    end1[ll] = curent_state

                # append ini
                list1 = []
                list1.append(ini)
                mm = np.ones(shape=(4, sizen))
                mm[0, :] = ini

            for i in range(4):

                if(i ==0):
                    ini = deepcopy(self.GLS_si(ini=deepcopy(ini),t=branch_length[i], sizen=sizen)[1])
                    leaf = deepcopy(self.GLS_si(ini=deepcopy(ini),t=branch_length[i+3], sizen=sizen)[1])
                    list.append(leaf)
                    list1.append(ini)
                    mm[i + 1, :] = ini


                elif(i ==1):
                    ini = deepcopy(self.GLS_si(ini=deepcopy(ini),t=branch_length[i+1], sizen=sizen)[1])
                    leaf = deepcopy(self.GLS_si(ini=deepcopy(ini),t=branch_length[i+4], sizen=sizen)[1])
                    list.append(leaf)
                    list1.append(ini)
                    mm[i + 1, :] = ini

                elif(i ==2):
                    ini = deepcopy(self.GLS_si(ini=deepcopy(ini),t=branch_length[i+2], sizen=sizen)[1])
                    leaf = deepcopy(self.GLS_si(ini=deepcopy(ini),t=branch_length[i+5], sizen=sizen)[1])
                    list.append(leaf)
                    list1.append(ini)
                    mm[i + 1, :] = ini

                else:
                    # ini is internal node, leaf is observed;
                    # list store observed
                    leaf = deepcopy(self.GLS_si(ini=deepcopy(ini), sizen=sizen, t=branch_length[7],ifdet=True)[1])
                    list.append(leaf)

                self.measure_difference(ini, leaf, 1)

            # append outgroup
            list.append(end1)

            save_nameP = '../test/savesample/RRR_Internal_' + geneconv.paralog[0] + geneconv.paralog[
                1] + 'sample.txt'

            mm = np.array(mm, order="F")
            np.savetxt(save_nameP, mm)

            #  with open(save_nameP, 'wb') as f:
            #    pickle.dump(mm, f)

            return list

    def gls_true(self,t=0.1):
        if self.Q_original is None:
            self.original_Q()

        print(self.Q_original)
        P = linalg.expm(self.Q_original * t)
        em=np.zeros(shape=(2, 4))

        pnew=[poisson.pmf(0, 2*t), poisson.pmf(1, 2*t), poisson.pmf(2, 2*t), 1 - poisson.cdf(2, 2*t)]


        ### compute  the k=0
        q1 = np.zeros(shape=(16, 16))
        em[0,0]=q1[0,0]=pnew[0]/P[0, 0]

        em[1,1]=pnew[1]/(1-P[1,1])

        em[1,2]=(14/15)*pnew[2]/(1-P[1,1])
        em[0, 2] = (1/15)/P[0, 0]

        print(em)

    def make_ini_testgls(self, sizen, half):
        ini = np.ones(sizen)
        z = np.ones(16)/16

        sample = np.arange(0, 16, 1)

        for i in range(sizen):
            ini[i] = int(np.random.choice(sample, 1, p=(z))[0])

        end = deepcopy(ini)
        for i in range(half):
            end[i] =copy.copy( int(np.random.choice(sample, 1, p=(z))[0]))

            while int(end[i]) == int(ini[i]):
                end[i] = int(np.random.choice(sample, 1, p=(z))[0])

        return ini,end

### test gls algorithm
    def test_gls(self,sizen=10,half=1):
       rr=self.make_ini_testgls(sizen=sizen,half=half)
       ini=rr[0]
       end=rr[1]

       ini=np.ones(sizen)*0
       end=np.ones(sizen)

       self.gls_true()


       print(self.GLS_m_test(t=0.2,ini=ini,end=end,sizen=sizen))

    def GLS_m_test(self, t=1, ini=None, end=None, sizen=40):
        global di
        global di1

        di = 16
        di1 = 9

        #### making ini

        if self.Q_new is None:
            self.making_Qg()

        Q_iiii = np.ones(di)
        for ii in range(di):
            Q_iiii[ii] = sum(self.Q[ii,])

        em = np.zeros(shape=(2, 5))


        time_list = []
        state_list = []
        dis_matrix = np.ones(sizen)

        for i in range(sizen):
            if ini[i] != end[i]:
                time_list.append(0)
                state_list.append(0)
                dis_matrix[i] = 0
            else:
                time_list.append(1)
                state_list.append(1)

        # start simulation

        for ii in range(sizen):

            # time_list[ii] ==0 means there is a mutation, initial state not equal to  end state
            # Q_iiii means a diagonal entries of rate matrix

            if time_list[ii] == 0:
                for jj in range(1):
                    # most transfer 10 times
                    current_state = ini[ii]
                    i = 0

                    while current_state != end[ii]:
                        current_state = ini[ii]
                        i = 1
                        state = [0]
                        u1 = np.random.uniform(0, 1)
                        u = -np.log((1 - (1 - np.exp(-Q_iiii[int(current_state)] * t)) * u1)) / \
                            (Q_iiii[int(current_state)])
                        a = np.random.choice(range(di1), 1, p=self.Q_new[int(current_state),])[0]
                        current_state = self.dic_col[int(current_state), a] - 1

                        # if jump to absorbing state and without finishing process, we need to resample

                        #    while sum(self.Q_new[int(current_state), ]) == 0:
                        #         a = np.random.choice(range(di1), 1, p=self.Q_new[int(old), ])[0]
                        #         current_state = self.dic_col[int(old), a] - 1
                        state.append(int(current_state))

                        while u <= t:
                            i = i + 1
                            u = u + random.exponential(1 / Q_iiii[int(current_state)])
                            a = np.random.choice(range(di1), 1, p=self.Q_new[int(current_state),])[0]
                            current_state = self.dic_col[int(current_state), a] - 1

                            # if jump to absorbing state and without finishing process, we need to resample

                            state.append(int(current_state))
                        current_state = state[i - 1]

                    print(i)
                    print(state)
                    print(state[0:i])


                    if (i == 1):
                        em[1, 1] = em[1, 1] + 1
                    elif (i == 2):
                        em[1, 2] = em[1, 2] + 1
                    elif (i == 3):
                        em[1, 3] = em[1, 3] + 1
                    else:
                        em[1, 4] = em[1, 4] + 1

#### end==ini
            else:
                for jj in range(1):
                    # most transfer 10 times
                    current_state = 11111
                    i=0


                    while current_state != end[ii]:
                        current_state = ini[ii]
                        i=0
                        u=0
                        state = [0]
                        while u <= t:
                            u = u + random.exponential(1 / Q_iiii[int(current_state)])
                            if u <= t:
                                i = i + 1
                                a = np.random.choice(range(di1), 1, p=self.Q_new[int(current_state),])[0]
                                current_state = self.dic_col[int(current_state), a] - 1
                                # if jump to absorbing state and without finishing process, we need to resample
                                state.append(int(current_state))
                    k1=0
                    if i>=1:
                        state[0]=ini[ii]
                        for k in range(i):
                            if((state[k]%4)!=(state[k+1]%4)):
                                k1=k1+1



                    if (i == 0):
                        em[0, 0] = em[0, 0] + 1
                    elif (i == 1):
                        em[0, 1] = em[0, 1] + 1
                    elif (i == 2):
                        em[0, 2] = em[0, 2] + 1
                    elif (i == 3):
                        em[0, 3] = em[0, 3] + 1
                    else:
                        em[0, 4] = em[0, 4] + 1

        return em

### measure difference of paralog
    def measure_difference(self,ini,end,branch):


        mutation_rate=0
        ini_paralog_div=0
        end_paralog_div = 0
        str = {0, 5, 10, 15}

        for i  in range(self.sites_length):
            if(ini[i]!=end[i]):
                mutation_rate=mutation_rate+1
            if(ini[i] in str):
                ini_paralog_div=ini_paralog_div+1
            if(end[i]in str):
                end_paralog_div=end_paralog_div+1

        print("for branch :", geneconv.edge_list[branch])
        print("estimated branch length :", self.scene['tree']["edge_rate_scaling_factors"][branch])
        print("%  identity between  paralogs at branch beginning:", ini_paralog_div/self.sites_length)
        print("%  identity between  paralogs at branch ending:", end_paralog_div / self.sites_length)
        print("%  sites differ between beginning and ending in at least one", mutation_rate/self.sites_length)
        print("**********************")

# special for EM
    def whether_IGC_EM(self,history_matrix,effect_number):

        if self.Model == "HKY":
            for ii in range(effect_number):

                i_b=int(history_matrix[ii, 7])//4
                j_b = int(history_matrix[ii, 7]) % 4
                i_p = int(history_matrix[ii, 8]) // 4
                j_p = int(history_matrix[ii, 8]) % 4

                if i_p == j_p:
                    if i_b != j_b and i_b == i_p:
                        # y_coor is corresponding coor for igc
                        y_coor = np.argwhere(self.dic_col[int(history_matrix[ii, 7]),] == (int(history_matrix[ii, 8]) + 1))[0]
                        qq=self.Q[int(history_matrix[ii, 7]), y_coor]
                        history_matrix[ii, 2]=(self.tau)/qq

                    elif(i_b!=j_b and j_b==j_p):
                        y_coor=np.argwhere(self.dic_col[int(history_matrix[ii, 7]),] == (int(history_matrix[ii, 8]) + 1))[0]
                        qq=self.Q[int(history_matrix[ii, 7]),y_coor]
                        history_matrix[ii, 2]=(self.tau)/qq

        return history_matrix


    def monte_carlo_s_EM(self,ifsave=True,
                    iftestsimulation=True,sizen=50000,times=1):

        self.change_t_Q(tau=0.1)
        self.sites_length=sizen
        list=[]

        if iftestsimulation==True:

            ttt = len(self.scene['tree']["column_nodes"])
            for j in range(ttt):
                t1 = 0.02
                print(j)

                if j == 0:
                    ini1 = self.make_ini(sizen=sizen)
                    re=self.GLS_si(t=t1,ini=ini1,sizen=sizen)
                    list.append(re[1])

             #       sam = self.rank_ts(time=re[2], t=t1, state=re[3], ini=ini1, effect_number=re[4])
             #       re1 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1])
             #       effect_number = re1[1]
              #      re1 = re1[0]
                    re1=None
                    effect_number=0

                elif j == 1:
                    print("ignore the outgroup")


                elif j >= 2 and j <=5:

                    ift=1

                    ini2 = self.geneconv.node_to_num[geneconv.edge_list[j][0]]
                    self.change_t_Q(tau=0.1)
                    ini1=deepcopy(list[int(ini2//2)])
                    re=self.GLS_si(t=t1,ini=ini1,sizen=sizen)
                    if j%2==0:
                        list.append(re[1])
                #        self.measure_difference(ini=list[int(ini2//2)],end=re[1],branch=1)

                    sam = self.rank_ts(time=re[2], t=t1, state=re[3], ini=re[0], effect_number=re[4])
                    self.change_t_Q(tau=0.1)
                    re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1],branch=j,times=0)
                    effect_number1 = deepcopy(re2[1])
                    re2 = deepcopy(re2[0])


                    di=0
                    for jj in range(sizen):
                            di=self.dic_di[re[1][jj]]+di
                    print(di)

                    if re1 is  None:
                        re1=deepcopy(re2)
                        ift=0
                        effect_number=deepcopy(effect_number1)

                    if ift !=0:
                       re1 = deepcopy(np.vstack((re1, re2)))
                       effect_number = effect_number1 + effect_number

                elif j >5:

                    ini2 = self.geneconv.node_to_num[geneconv.edge_list[6][0]]
                    self.change_t_Q(tau=0.1)
                    ini1=deepcopy(list[int(ini2//2)])
                    re=self.GLS_si(t=t1,ini=ini1,sizen=sizen)


                    sam = self.rank_ts(time=re[2], t=t1, state=re[3], ini=re[0], effect_number=re[4])
                    self.change_t_Q(0.1)
                    re2 = self.whether_IGC(history_matrix=sam[0], effect_number=sam[1],branch=j,times=0)
                    effect_number1 = deepcopy(re2[1])
                    re2 = deepcopy(re2[0])
                    print(effect_number1)


                    di=0
                    for jj in range(sizen):
                            di=self.dic_di[re[1][jj]]+di
                    print(di)


                    re1 = deepcopy(np.vstack((re1, re2)))
                    effect_number = effect_number1 + effect_number




        if ifsave==True:

           save_nameP = '../test/savesample/Ind_' + geneconv.Model+ geneconv.paralog[0]+geneconv.paralog[1]+'simulation.txt'
           np.savetxt(open(save_nameP, 'w+'), re1.T)

        return re1 , effect_number





if __name__ == '__main__':
    #
    # paralog = ['EDN', 'ECP']
    # alignment_file = '../test/EDN_ECP_Cleaned.fasta'
    # newicktree = '../test/EDN_ECP_tree.newick'
    # name = 'EDN_ECP_full'

    # paralog = ['paralog0', 'paralog1']
    # alignment_file = '../test/fixtau_ednecp_real.fasta'
    # newicktree = '../test/sample1.newick'
    # name = 'fixtau_ednecp_real1'

    paralog = ['__Paralog1', '__Paralog2']
    alignment_file = '../test/intron/group_972_intron2_c.fasta'
    newicktree = '../test/intron/intronc.newick'
    name ="intron_972_2_c"

 #   name = 'tau99_01vss'
  #  Force ={0:np.exp(-0.71464127), 1:np.exp(-0.55541915), 2:np.exp(-0.68806275),3: np.exp( 0.74691342),4: np.exp( -0.5045814)}
    # %AG, % A, % C, kappa, tau
    #Force= {0:0.5,1:0.5,2:0.5,3:1,4:0}
    Force=None
    model = 'HKY'

    type='situation1'
    save_name = '../test/save/' + model + name+'_'+type+'_nonclock_save.txt'
    geneconv = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name)

    self = AncestralState(geneconv)
    scene = self.get_scene()



 #   self.jointly_common_ancstral_inference()
  #  print(self.geneconv.node_to_num)
  #  print(geneconv.edge_list)



########test common ancter wt
  #  self.test_pro11(node_s=[0, 0, 0, 12], site_s=0,mc=5000)
    #print(self.geneconv.edge_to_blen)
    #print(np.exp(self.geneconv.x_rates))



#####################################################
######generit simulation data
#####################################################
    #
    # sizen=20000
    # self.change_t_Q(tau=1)
    # aaa=self.topo_EDNECP(sizen=sizen)
    # self.difference(ini=aaa,sizen=sizen)
    # print(self.trans_into_seq(ini=aaa,sizen=sizen))

#####################################################
######generit simulation data
#####################################################

########test common ancter whether work
    # self.jointly_common_ancstral_inference()


## method "simple" is default method， which focus on quail from post dis
## method "divide" is using the biggest difference among paralogs, and make category
#####################################
################TEST
######################################
    self.get_igcr_pad(times=30, repeat=1,simple_state_number=8, method="simple",EM_circle=5)
 #   print(self.get_parameter(function="exp"))
 #   self.MC_EM(times=1, repeat=1, ifpermutation=False, ifwholetree=True, ifsave=False, method="bybranch",EM_circle=10)
  #  self.get_igcr_pad(times=10, repeat=1, ifpermutation=False, ifwholetree=True, ifsave=False, method="bybranch")
# print(self.get_parameter(function="linear"))
#####################################
################TEST
####################################

   # print(11111111111111)
    #print(self.get_parameter(function="squre"))
    #print(self.get_igcr_pad(times=1, repeat=1,ifpermutation=False,ifwholetree=False,ifsave=True,method="divide"))

    # print(self.node_length)

    # print(geneconv.edge_to_blen)
    # print(geneconv.num_to_node)
    # print(geneconv.edge_list)
    # print(scene['tree'])
    #

    #
    #
    #self.rank_ts()
       # self.monte_carlo(times=2,repeat=2)
        # print(self.igc_com)

    #aa=self.monte_carlo(times=1,repeat=2)



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
#     for i in range(len(j_out["responses"][0][0])):
#         print(j_out["responses"][0][0][i]1)
#         aa=array(j_out["responses"][0][0][i])+aa
#     aa=self.get_scene()
#     print(aa["observed_data"])
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