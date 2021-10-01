# coding=utf-8
# A separate file for Ancestral State Reconstruction
#output for GLM
# Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function
import jsonctmctree.ll, jsonctmctree.interface
from IGCexpansion.CodonGeneconv import *
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from numpy import random
from scipy import linalg
import copy
from scipy.stats import poisson

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


class AncestralState1:

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
        self.ifXiang=True

        self.name=self.geneconv.save_name

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
        scene=self.get_scene()

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

    def isNonsynonymous(self, ca, cb, codon_table):
        return (codon_table[ca] != codon_table[cb])


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
     #   print(self.scene["observed_data"]["idiverge_observations"])
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
            diverge=1
            for j in range(tree_len-1):
                if(self.geneconv.node_to_num[geneconv.edge_list[j][0]]==inode and diverge==1):
                    diverge=diverge+1
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


    def difference(self,ini):
        index = 0
        ratio_nonsynonymous = 0
        ratio_synonymous = 0


        if self.Model=="HKY":
            str = {0, 5, 10, 15}

            for i in range(self.sites_length):
                if not ini[i] in str:
                    index=index+1
        else:
            for i in range(self.sites_length):
                ca=(ini[i])//61
                cb=(ini[i])%61
                if ca != cb:
                    index=index+1
                    cb1=self.geneconv.state_to_codon[cb]
                    ca1 = self.geneconv.state_to_codon[ca]
                    if self.isNonsynonymous(cb1, ca1, self.geneconv.codon_table):
                        ratio_nonsynonymous=ratio_nonsynonymous+1

        if not index==0:
            ratio_nonsynonymous = ratio_nonsynonymous/index
            ratio_synonymous = 1 - ratio_synonymous



        print(index)
        print(ratio_nonsynonymous)
        print("xxxxxxxxxxxx")


        return index,ratio_nonsynonymous,ratio_synonymous



    def get_paralog_diverge(self,repeat=10):
    #    print(self.geneconv.codon_nonstop)
      #  self.geneconv.get_MG94Geneconv_and_MG94()


        list=[]
        list1=[]
        list.append(self.sites_length)
        list.append(self.geneconv.tau)
        list.append(self.geneconv.kappa)
        if self.Model == "MG94":
            list.append(self.geneconv.omega)

        diverge_list=[]
        diverge_listnonsynonymous=[]
        diverge_listsynonymous=[]


        self.geneconv.get_ExpectedNumGeneconv()
        tau=self.geneconv.get_summary(branchtau=True)
        ttt = len(self.scene['tree']["column_nodes"])

        for mc in range(repeat):

            self.jointly_common_ancstral_inference()
            for j in range(ttt):

                            if self.Model=="HKY":
                                if not j == 1:
                                     ini2 = self.geneconv.node_to_num[self.geneconv.edge_list[j][0]]
                                     end2 = self.geneconv.node_to_num[self.geneconv.edge_list[j][1]]
                                     ini1 = deepcopy(self.sites[ini2,])
                                     end1 = deepcopy(self.sites[end2,])

                                     diverge=(self.difference(ini1)[0]+self.difference(end1)[0])*0.5
                                if mc != 0:
                                     diverge_list[j] = diverge_list[j] + diverge
                                else:
                                     diverge_list.append(diverge)
                            if self.Model=="MG94":
                                if not j == 1:
                                     ini2 = self.geneconv.node_to_num[self.geneconv.edge_list[j][0]]
                                     end2 = self.geneconv.node_to_num[self.geneconv.edge_list[j][1]]
                                     ini1 = deepcopy(self.sites[ini2,])
                                     end1 = deepcopy(self.sites[end2,])

                                     ini_D=self.difference(ini1)[0]
                                     end_D = self.difference(end1)[0]
                                     ini_ratio_nonsynonymous=self.difference(ini1)[1]
                                     end_ratio_nonsynonymous = self.difference(end1)[1]
                                     ini_ratio_synonymous = self.difference(ini1)[2]
                                     end_ratio_synonymous = self.difference(end1)[2]
                                     diverge_nonsynonymous = (ini_D*ini_ratio_nonsynonymous+end_D*end_ratio_nonsynonymous)*0.5
                                     diverge_synonymous = (ini_D * ini_ratio_synonymous + end_D * end_ratio_synonymous) * 0.5

                                     diverge = float(ini_D + end_D) * 0.5

                                if mc != 0:
                                    diverge_listnonsynonymous[j] = diverge_listnonsynonymous[j] + diverge_nonsynonymous
                                    diverge_listsynonymous[j] = diverge_listsynonymous[j] + diverge_synonymous
                                    diverge_list[j] = diverge_list[j] + diverge
                                else:
                                    diverge_listnonsynonymous.append(diverge_nonsynonymous)
                                    diverge_listsynonymous.append(diverge_synonymous)
                                    diverge_list.append(diverge)





        for j in range(ttt):
                    list.append(tau[0][j])

#        print(divergelist)
        if self.Model == "HKY":
              for j in range(ttt):
                    list.append(diverge_list[j]/repeat)
        elif self.Model == "MG94":
              for j in range(ttt):
                    list.append(diverge_listnonsynonymous[j]/repeat)
              for j in range(ttt):
                    list.append(diverge_listsynonymous[j]/repeat)
              for j in range(ttt):
                    d=float(diverge_list[j])/repeat
                    list.append(d)

      #  print(tau[1])
        #exoect igc

        for j in self.geneconv.edge_list:
                list.append(tau[1][j])

#     branch length
        for j in self.geneconv.edge_list:
                 list.append(tau[2][j])

# opportunity time
        for j in self.geneconv.edge_list:
                list.append(tau[3][0][j])
        for j in self.geneconv.edge_list:
                list.append(tau[3][1][j])

        list1.extend([("brahch",a, b) for (a, b) in self.geneconv.edge_list])



      #  print(list)


        save_nameP = "./save/"+self.name +'.txt'
        with open(save_nameP, 'w+') as f:
            np.savetxt(f, list, delimiter=' ', )


        save_nameP1 = "./save/"+self.name +'.name.txt'
        np.savetxt(save_nameP1, list1,fmt="%s")


    def print(self):
        self.geneconv.get_ExpectedNumGeneconv()






if __name__ == '__main__':


    # group_693_intron3 group_317_intron1 is too short
 #    name = "group_972_intron4_c"
 #
 #    paralog = ['__Paralog1', '__Paralog2']
 #    alignment_file = '../test/intron/'+name+'.fasta'
 #    newicktree = '../test/intron/intronc.newick'
 #
 # #   name = 'tau99_01vss'
 #  #  Force ={0:np.exp(-0.71464127), 1:np.exp(-0.55541915), 2:np.exp(-0.68806275),3: np.exp( 0.74691342),4: np.exp( -0.5045814)}
 #    # %AG, % A, % C, kappa, tau
 #    #Force= {0:0.5,1:0.5,2:0.5,3:1,4:0}
 #    Force=None
 #    model = 'HKY'
 #
 #    type='situation1'
 #    save_name = name
 #    geneconv = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
 #                               save_path='../test/save/', save_name=name)
 #
 #    self = AncestralState(geneconv)
 #    scene = self.get_scene()
 #    self.get_paralog_diverge()

    name = "YBL087C_YER117W_input"

    paralog = ['YBL087C', 'YER117W']
    alignment_file = '../test/yeast/' + name + '.fasta'
    newicktree = '../test/yeast/YeastTree.newick'

    #   name = 'tau99_01vss'
    #  Force ={0:np.exp(-0.71464127), 1:np.exp(-0.55541915), 2:np.exp(-0.68806275),3: np.exp( 0.74691342),4: np.exp( -0.5045814)}
    # %AG, % A, % C, kappa, tau
    # Force= {0:0.5,1:0.5,2:0.5,3:1,4:0}
    Force = None
    model = 'MG94'

    type = 'situation_new'
    save_name = model+name
    geneconv = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name)

    self = AncestralState1(geneconv)
    print(([(None, None)] * (6 - 4)))


 #   scene = self.get_scene()

 #   print(scene['tree']['edge_processes'])

   # self.get_paralog_diverge()
  #  print(geneconv.omega)
  #  print(geneconv.IGC_Omega)
