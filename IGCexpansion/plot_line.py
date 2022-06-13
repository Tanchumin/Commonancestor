
from __future__ import print_function, absolute_import
from copy import deepcopy
from CodonGeneconFunc import *
from CodonGeneconv import *
import numpy as np

from scipy import linalg
from IGCexpansion.CodonGeneconFunc import isNonsynonymous

class plot_line:

    def __init__(self,
                 geneconv  # JSGeneconv analysis for now
                 ):
        self.geneconv = geneconv
        self.kappa=1.0
        self.pi=[0.25, 0.25, 0.25, 0.25]
        self.prior_distribution=[0.25, 0.25, 0.25, 0.25]
        self.tau=0
        self.ifmodel = False
        self.K=1.1
        self.Model = 'HKY'
        self.Q =None
        self.dic_col = None
        self.Q_original=None
        self.time=0


    def get_HKYBasic(self):
            Qbasic = np.array([
                [0, 1.0, self.kappa, 1.0],
                [1.0, 0, 1.0, self.kappa],
                [self.kappa, 1.0, 0, 1.0],
                [1.0, self.kappa, 1.0, 0],
            ]) * np.array(self.pi)
            if np.dot(self.prior_distribution, Qbasic.sum(axis=1)) != 0:
                expected_rate = np.dot(self.prior_distribution, Qbasic.sum(axis=1))
            else:
                expected_rate = 1
            Qbasic = Qbasic / expected_rate
            return Qbasic

    def Get_branch_QHKY(self,paralogid=0.95):

        Qbasic = self.get_HKYBasic()

        Qlist = []

        if self.ifmodel==False:

            row = []
            col = []

            rate_geneconv = []
            for i, pair_from in enumerate(product('ACGT', repeat=2)):
                na, nb = pair_from
                sa = self.genecove.nt_to_state[na]
                sb =  self.genecove.nt_to_state[nb]
                for j, pair_to in enumerate(product('ACGT', repeat=2)):
                    nc, nd = pair_to
                    sc =  self.genecove.nt_to_state[nc]
                    sd =  self.genecove.nt_to_state[nd]
                    if i == j:
                        continue
                    GeneconvRate = get_HKYGeneconvRate(pair_from, pair_to, Qbasic,
                                                       self.tau)
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

                  row = []
                  col = []


                  rate_geneconv = []
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
                          GeneconvRate = get_HKYGeneconvRate(pair_from, pair_to, Qbasic, self.tau*np.power(paralogid, self.K))
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

        return Qlist


    def making_Qmatrix(self):

        if self.Model=="HKY":
           Q_trans=self.Get_branch_QHKY()[2]

        else:
           Q_trans = self.Get_branch_QHKY()[2]

        scene = self.geneconv.get_scene()

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

    def compute(self):

        self.original_Q(if_rebuild_Q=True)
        Q = linalg.expm(self.Q_original * self.time)


