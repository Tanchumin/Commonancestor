
# A separate file for Ancestral State Reconstruction
# Uses Alex Griffing's JsonCTMCTree package for likelihood and gradient calculation
# Xiang Ji
# xji3@ncsu.edu
# Tanchumin Xu
# txu7@ncsu.edu
from __future__ import print_function
import jsonctmctree.ll, jsonctmctree.interface
from JSGeneconv import JSGeneconv
from CodonGeneconv import *
from Func import *
from copy import deepcopy
import os
from Common import *
import numpy as np


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
        self.node_length=0
        self.sites_length = self.geneconv.nsites
        self.Model=self.geneconv.Model
        self.ifmarginal = False

        if isinstance(geneconv, JSGeneconv):
            raise RuntimeError('Not yet implemented!')


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


    def get_maxpro_index(self,ifmarginal=False,paralog=1):

        if self.ancestral_state_response is None:
            self.ancestral_state_response = self.get_ancestral_state_response()

        if ifmarginal==False:
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
            else:
                self.node_length = len(self.get_num_to_node())
                sites = np.zeros(shape=(self.node_length, self.sites_length))
                for node in range(self.node_length):
                    mat=self.get_marginal(node,paralog)
                    for site in range(self.sites_length):
                        sites[node][site] = np.argmax(np.array(mat)[site,:])

        return (sites)

    def get_maxpro_matrix(self,ifmarginal=False,paralog=1):

        if self.ancestral_state_response is None:
            self.ancestral_state_response = self.get_ancestral_state_response()

        if ifmarginal==False:
            self.node_length=len(self.get_num_to_node())
            sites = np.zeros(shape=(self.node_length,self.sites_length ))
            for site in range(self.sites_length):
                for node in range(self.node_length):
                    sites[node][site] = np.max(np.array(self.ancestral_state_response[site])[:, node])

        else:
            if paralog==1:
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

    def get_marginal(self,node,paralog=1):
        if self.ancestral_state_response is None:
            self.ancestral_state_response = self.get_ancestral_state_response()

        if paralog==1:

            if self.Model=='MG94':
                marginal_sites = np.zeros(shape=(self.sites_length,61))
                for site in range(self.sites_length):
                    i=0
                    for marginal in range(61):
                        marginal_sites[site][marginal] = sum(np.array(self.ancestral_state_response[site])[i:(i+61), node])
                        i=i+61

            else:
                marginal_sites = np.zeros(shape=(self.sites_length, 4))
                for site in range(self.sites_length):
                    i = 0
                    for marginal in range(4):
                        marginal_sites[site][marginal] = sum(np.array(self.ancestral_state_response[site])[i:(i + 4), node])
                        i = i + 4
        else:
            if self.Model == 'MG94':
                marginal_sites = np.zeros(shape=(self.sites_length, 61))
                for site in range(self.sites_length):
                    i = 0
                    for marginal in range(61):
                        index_pra2=range(i,3671+i,61)
                        marginal_sites[site][marginal] = sum(np.array(self.ancestral_state_response[site])[index_pra2, node])
                        i = i + 1

            else:
                marginal_sites = np.zeros(shape=(self.sites_length, 4))
                for site in range(self.sites_length):
                    i = 0
                    for marginal in range(4):
                        index_pra2=range(i,i+16,4)
                        marginal_sites[site][marginal] = sum(np.array(self.ancestral_state_response[site])[index_pra2, node])
                        i = i + 1


        return marginal_sites

    def get_interior_node(self):
        if self.node_length is None:
             self.node_length= len(self.get_num_to_node())

        node = np.arange(self.node_length)
        interior_node = set(node) - set(self.scene["observed_data"]["nodes"])
        c = [i for i in interior_node]
        return (c)


if __name__ == '__main__':

    paralog = ['EDN', 'ECP']
    alignment_file = '../test/EEEE.fasta'
    newicktree = '../test/EEEE.newick'
    Force = {5:0,6:0,7:1}
    Force1=None
    model = 'MG94'


    name='EDN_ECP'
    type='situation5'
    save_name = '../test/save/' + model + name+'_'+type+'_nonclock_save.txt'
    save_name1 ='../test/save/In3_MG94_EDN_ECP_nonclock_save110.txt'
    geneconv = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name)
    test = AncestralState(geneconv)
    self = test

    print(test.geneconv.codon_to_state)
    scene = self.get_scene()


    save=self.get_maxpro_matrix(True,1)
    save_namep = '../test/savecommon32/Ind_' + model + '_1_'+type+'_'+name+'_maxpro.txt'
    np.savetxt(open(save_namep, 'w+'), save.T)


    save=self.get_maxpro_matrix(True,2)
    save_namep = '../test/savecommon32/Ind_' + model + '_2_'+type+'_'+name+'_maxpro.txt'
    np.savetxt(open(save_namep, 'w+'), save.T)

    save=self.get_maxpro_index(True,1)
    save_namep = '../test/savecommon32/Ind_' + model + '_1_'+type+'_'+name+'_maxproind.txt'
    np.savetxt(open(save_namep, 'w+'), save.T)

    save=self.get_maxpro_index(True,2)
    save_namep = '../test/savecommon32/Ind_' + model + '_2_'+type+'_'+name+'_maxproind.txt'
    np.savetxt(open(save_namep, 'w+'), save.T)

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


