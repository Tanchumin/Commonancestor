# coding=utf-8
# A separate file for Ancestral State Reconstruction
#output for GLM
# Tanchumin Xu
# txu7@ncsu.edu

from __future__ import print_function
import jsonctmctree.ll, jsonctmctree.interface
from IGCexpansion.CodonGeneconv import *
from copy import deepcopy
import numpy as np




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

        self.sites_length = self.geneconv.nsites
        self.Model = self.geneconv.Model

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


        self.node_length=0
        self.ifmarginal = False
        self.process=self.geneconv.processes

        self.min_diff=0

# relationship is a matrix about igc rates on different paralog
# igc_com is matrix contain paralog difference,difference time, igc state, paralog category
        self.relationship=None
        self.igc_com=None
        self.judge=None
        self.ifmax=False

        self.name=self.geneconv.save_name

        self.dwell_id=True
        self.scene_ll=None
        self.ifDNA=True

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
            if j_out['status'] == 'feasible':
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
            if j_out['status'] == 'feasible':
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
        promax=self.get_maxpro_index(ifmarginal=ifmarginal,paralog=paralog)
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


# compute Q matrix  with 0


    def jointly_common_ancstral_inference(self,ifcircle=False,taulist=None):


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



        return index,ratio_nonsynonymous,ratio_synonymous



    def get_paralog_diverge(self,repeat=10,ifrobust=False):
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
        ttt = len(self.geneconv.tree['col'])

        diverge_listnonsynonymous = np.zeros(ttt)
        diverge_listsynonymous = np.zeros(ttt)


        self.geneconv.get_ExpectedNumGeneconv()
        tau=self.geneconv.get_summary(branchtau=True,robust=ifrobust)
        ttt = len(self.scene['tree']["column_nodes"])

        if self.dwell_id == True:

            expected_DwellTime = self._ExpectedHetDwellTime()
            if self.Model == "MG94":
                if self.ifDNA == True:
                    expected_DwellTime = self._ExpectedHetDwellTime_DNA()

                    id = [1 - (((
                                        expected_DwellTime[0][i] + expected_DwellTime[1][i]) +
                                (expected_DwellTime[2][i] + expected_DwellTime[3][i]) * 2 +
                                (expected_DwellTime[4][i] + expected_DwellTime[5][i]) * 3)
                               / (2 * 3 * self.geneconv.nsites))
                          for i in range(ttt)]

                    diverge_listnonsynonymous = [1 - (((
                                                           expected_DwellTime[1][i]) +
                                                       (expected_DwellTime[3][i]) * 2 +
                                                       (expected_DwellTime[5][i]) * 3)
                                                      / (2 * 3 * self.geneconv.nsites))
                                                 for i in range(ttt)]
                    diverge_listsynonymous = [1 - (((
                                                        expected_DwellTime[0][i]) +
                                                    (expected_DwellTime[2][i]) * 2 +
                                                    (expected_DwellTime[4][i]) * 3)
                                                   / (2 * 3 * self.geneconv.nsites))
                                              for i in range(ttt)]



                else:
                    expected_DwellTime = self._ExpectedHetDwellTime()
                    id = [1 - ((
                                       expected_DwellTime[0][i] + expected_DwellTime[1][i])
                               / (2 * self.geneconv.nsites))
                          for i in range(ttt)]

                    diverge_listnonsynonymous = [1 - ((
                                                          expected_DwellTime[1][i])
                                                      / (2 * self.geneconv.nsites))
                                                 for i in range(ttt)]
                    diverge_listsynonymous = [1 - ((
                                                       expected_DwellTime[0][i])
                                                   / (2 * self.geneconv.nsites))
                                              for i in range(ttt)]







            else:
                id = [1 - (expected_DwellTime[i] / (2 * self.geneconv.nsites
                                                    ))
                      for i in range(ttt)]


            if self.Model == "HKY":
                  for j in range(ttt):
                        list.append(id[j])
            elif self.Model == "MG94":
                  for j in range(ttt):
                        list.append(diverge_listnonsynonymous[j])
                  for j in range(ttt):
                        list.append(diverge_listsynonymous[j])
                  for j in range(ttt):
                        list.append(id[j])


        else:
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

           if self.Model == "HKY":
                for j in range(ttt):
                    list.append(diverge_list[j] / repeat)
           elif self.Model == "MG94":
                for j in range(ttt):
                    list.append(diverge_listnonsynonymous[j] / repeat)
                for j in range(ttt):
                    list.append(diverge_listsynonymous[j] / repeat)
                for j in range(ttt):
                    d = float(diverge_list[j]) / repeat
                    list.append(d)


        # branch tau
        for j in range(ttt):
                    list.append(tau[0][j])

#        print(divergelist)


      #  print(tau[1])
        #exoect igc

        if ifrobust==False:
            for j in self.geneconv.edge_list:
                    list.append(tau[1][j])
        else:
           for j in range(len(self.geneconv.edge_list)):
                   list.append(tau[1][j])

#     branch length
        for j in self.geneconv.edge_list:
                 list.append(tau[2][j])

# dwell time
        if self.Model == "MG94":
            for j in self.geneconv.edge_list:
                    list.append(tau[3][0][j])
            for j in self.geneconv.edge_list:
                    list.append(tau[3][1][j])
            for j in self.geneconv.edge_list:
                    list.append(tau[4][0][j])
            for j in self.geneconv.edge_list:
                     list.append(tau[4][1][j])

        else:
            for j in self.geneconv.edge_list:
                   list.append(tau[3][0][j])



        print(list)



        list1.extend([("brahch",a, b) for (a, b) in self.geneconv.edge_list])

        save_nameP = "./save/" + self.Model + "new_tau_paralog" + '.txt'
        with open(save_nameP, 'wb') as f:
           np.savetxt(f, list)




    def print1(self):
        print(self.geneconv.get_ExpectedNumGeneconv())

    def isSynonymous(self, first_codon, second_codon):
        return self.codon_table[first_codon] == self.codon_table[second_codon]




    def _ExpectedHetDwellTime(self, package='new', display=False):

        if package == 'new':
            self.scene_ll = self.get_scene()
            if self.Model == 'MG94':
                syn_heterogeneous_states = [(a, b) for (a, b) in
                                            list(product(range(len(self.geneconv.codon_to_state)), repeat=2)) if
                                            a != b and self.isSynonymous(self.geneconv.codon_nonstop[a], self.geneconv.codon_nonstop[b])]
                nonsyn_heterogeneous_states = [(a, b) for (a, b) in
                                               list(product(range(len(self.geneconv.codon_to_state)), repeat=2)) if
                                               a != b and not self.isSynonymous(self.geneconv.codon_nonstop[a],
                                                                                self.geneconv.codon_nonstop[b])]
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
                heterogeneous_states = [(a, b) for (a, b) in list(product(range(len(self.geneconv.nt_to_state)), repeat=2)) if
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

            ttt=len(self.geneconv.edge_list)
            if self.Model=="MG94":
                ExpectedDwellTime=np.zeros((2,ttt))

                for i in range(len(self.geneconv.edge_list)):
                    for j in range(2):
                        ExpectedDwellTime[j][i]=j_out['responses'][j][i]
            else:
                ExpectedDwellTime = np.zeros(ttt)

                for i in range(len(self.geneconv.edge_list)):
                     ExpectedDwellTime[i] = j_out['responses'][0][i]


            return ExpectedDwellTime
        else:
            print('Need to implement this for old package')

    def measure_difference_two(self,a,b):
        index=0

        for i in range(3):
            if a[i]!=b[i]:
                index=index+1

        return index

    def _ExpectedHetDwellTime_DNA(self, package='new', display=False):

        if package == 'new':
            self.scene_ll = self.get_scene()
            if self.Model == 'MG94':
                syn_heterogeneous_states_1 = [(a, b) for (a, b) in
                                            list(product(range(len(self.geneconv.codon_to_state)), repeat=2)) if
                                            a != b and self.isSynonymous(self.geneconv.codon_nonstop[a], self.geneconv.codon_nonstop[b]) and
                                            self.measure_difference_two(self.geneconv.codon_nonstop[a],
                                                                                self.geneconv.codon_nonstop[b])==1]
                nonsyn_heterogeneous_states_1 = [(a, b) for (a, b) in
                                               list(product(range(len(self.geneconv.codon_to_state)), repeat=2)) if
                                               a != b and not self.isSynonymous(self.geneconv.codon_nonstop[a],
                                                                                self.geneconv.codon_nonstop[b]) and
                                               self.measure_difference_two(self.geneconv.codon_nonstop[a],
                                                                                self.geneconv.codon_nonstop[b])==1]
                syn_heterogeneous_states_2 = [(a, b) for (a, b) in
                                              list(product(range(len(self.geneconv.codon_to_state)), repeat=2)) if
                                              a != b and self.isSynonymous(self.geneconv.codon_nonstop[a],
                                                                           self.geneconv.codon_nonstop[b]) and
                                              self.measure_difference_two(self.geneconv.codon_nonstop[a],
                                                                          self.geneconv.codon_nonstop[b]) == 2]
                nonsyn_heterogeneous_states_2 = [(a, b) for (a, b) in
                                                 list(product(range(len(self.geneconv.codon_to_state)), repeat=2)) if
                                                 a != b and not self.isSynonymous(self.geneconv.codon_nonstop[a],
                                                                                  self.geneconv.codon_nonstop[b]) and
                                                 self.measure_difference_two(self.geneconv.codon_nonstop[a],
                                                                             self.geneconv.codon_nonstop[b]) == 2]
                syn_heterogeneous_states_3 = [(a, b) for (a, b) in
                                              list(product(range(len(self.geneconv.codon_to_state)), repeat=2)) if
                                              a != b and self.isSynonymous(self.geneconv.codon_nonstop[a],
                                                                           self.geneconv.codon_nonstop[b]) and
                                              self.measure_difference_two(self.geneconv.codon_nonstop[a],
                                                                          self.geneconv.codon_nonstop[b]) == 3]
                nonsyn_heterogeneous_states_3 = [(a, b) for (a, b) in
                                                 list(product(range(len(self.geneconv.codon_to_state)), repeat=2)) if
                                                 a != b and not self.isSynonymous(self.geneconv.codon_nonstop[a],
                                                                                  self.geneconv.codon_nonstop[b]) and
                                                 self.measure_difference_two(self.geneconv.codon_nonstop[a],
                                                                             self.geneconv.codon_nonstop[b]) == 3]
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
                heterogeneous_states = [(a, b) for (a, b) in list(product(range(len(self.geneconv.nt_to_state)), repeat=2)) if
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

            ttt=len(self.geneconv.edge_list)
            if self.Model=="MG94":
                ExpectedDwellTime=np.zeros((6,ttt))

                for i in range(ttt):
                    for j in range(6):
                        ExpectedDwellTime[j][i]=j_out['responses'][j][i]
            else:
                ExpectedDwellTime = np.zeros(ttt)

                for i in range(len(self.geneconv.edge_list)):
                     ExpectedDwellTime[i] = j_out['responses'][0][i]



            return ExpectedDwellTime
        else:
            print('Need to implement this for old package')








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
    model = 'HKY'

    type = 'situation_new'
    save_name = model+name
    geneconv = ReCodonGeneconv(newicktree, alignment_file, paralog, Model=model, Force=Force, clock=None,
                               save_path='../test/save/', save_name=save_name)

    self = AncestralState1(geneconv)

    scene = self.get_scene()
    self.get_paralog_diverge()
