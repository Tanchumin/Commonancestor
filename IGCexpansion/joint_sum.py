#! /usr/bin/python3
# coding=utf-8

import multiprocessing as mp
from IGCexpansion.em_pt import *
import numdifftools as nd
from multiprocessing import Pool


class JointAnalysis_nest:
    auto_save_step = 2

    def __init__(self,
                 alignment_file_list,
                 tree_newick,
                 paralog_list,
                 Model = 'MG94',
                 IGC_Omega = None,
                 multiprocess_combined_list = None,
                 nnsites = None,
                 Force = None,
                 Shared = None,
                 save_path = './save/',
                 save_name = None,
                 post_dup = 'N1',
                 Force_share=None,
                 shared_parameters_for_k=[5,6],
                 Force_share_k={5: 0, 6: 0},
                 inibranch=0.1,
                 kini=3.1,
                 tauini=0.4,
                 omegaini=0.5):
        # first check if the length of all input lists agree
        assert (len(set([len(alignment_file_list), len(paralog_list)])) == 1)
        # doesn't allow IGC-specific omega for HKY model
        assert(Model == 'MG94' or IGC_Omega is None)

        self.save_path     = save_path
        self.Model         = Model
        self.IGC_Omega     = IGC_Omega
        self.paralog_list  = paralog_list
        self.x             = None
# this used for each data
        self.Force_share         = Force_share
        self.multiprocess_combined_list = multiprocess_combined_list


        #share k version:
        self.ifmodel = "old"

        if Shared is None:
            self.shared_parameters = []
        else:
            self.shared_parameters = Shared

        grand_save_name, individual_save_names = self.get_save_file_names(save_name)
        self.geneconv_list = [Embrachtau(tree_newick = tree_newick, alignment = alignment_file_list[i], paralog = paralog_list[i],
                                              Model = Model,  nnsites = nnsites,
                                              clock = False, Force = Force, save_path = save_path, save_name = individual_save_names[i],
                                              post_dup = post_dup,ifmodel="old",inibranch=inibranch,kini=kini,tauini=tauini,joint=True)
                              for i in range(len(alignment_file_list))]
        self.save_name     = grand_save_name

        self.auto_save = 0
        self.auto_save1 = 0

        self.initialize_x()
        self.shared_parameters_for_k=shared_parameters_for_k
        self.Force_share_k=Force_share_k


        self.fixtau=np.zeros(len(self.paralog_list))
        self.fixomega=np.zeros(len(self.paralog_list))
        self.fixk = np.zeros(len(self.paralog_list))
        self.siteslist=np.zeros(len(self.paralog_list))

        for i in range(len(self.paralog_list)):
            self.fixtau[i] = tauini
            self.fixk[i] = kini
            self.siteslist[i] = i
            if self.Model == "MG94":
                  self.fixomega[i]=omegaini

        self.ifexp=False


    def initialize_x(self):

            self.save_name1 = self.get_save_file_names(None)[0]
            self.shared_parameters = self.shared_parameters_for_k
            self.Force_share=self.Force_share_k


            if os.path.isfile(self.save_name1):
                for i in range(len(self.paralog_list)):
                       self.geneconv_list[i].renew_em_joint()
                self.initialize_by_save(self.save_name1)
                print('Successfully loaded parameter value from ' + self.save_name1)
                for i in range(len(self.paralog_list)):
                    self.fixk[i] = self.x[-1]


            self.update_by_x(deepcopy(self.x))
            if self.multiprocess_combined_list is None:
               self.multiprocess_combined_list = range(len(self.geneconv_list))

    def get_save_file_names(self, save_name):
        if len(self.shared_parameters):
            model_string = self.Model + '_withSharing'
        else:
            model_string = self.Model

        if save_name is None:

            general_save_name = self.save_path + 'Joint_' + model_string + '_' + str(len(self.paralog_list)) + '_pairs_grand_save.txt'

        else:
            general_save_name = save_name

        if self.ifmodel != "old":
            general_save_name = self.save_path + 'Joint_k_' + model_string + '_' + str(
                len(self.paralog_list)) + '_pairs_grand_save.txt'


        names = []
        for paralog in self.paralog_list:
            single_save_name = general_save_name.replace(str(len(self.paralog_list)) + '_pairs', '_'.join(paralog)).replace('_grand', '')
            names.append(single_save_name)

        return general_save_name, names

    def check_x_dim(self):
        assert(len(self.geneconv_list) > 1)
        if self.shared_parameters is None:
            shared_dim = 0
        else:
            shared_dim = len(self.shared_parameters)
        x_dim = sum([len(geneconv.x) for geneconv in self.geneconv_list]) - (len(self.geneconv_list) - 1) * shared_dim
        assert(len(self.x) == x_dim)

    def combine_x(self, uniq_x, shared_x):
        uniq_idx = 0
        shared_idx = 0
        x = []
        for i in range(len(self.geneconv_list[0].x)):
            if i in self.shared_parameters:
                x.append(shared_x[shared_idx])
                shared_idx += 1
            else:
                x.append(uniq_x[uniq_idx])
                uniq_idx += 1
        return x

    def update_by_x(self, x):
      #  self.check_x_dim()
        self.x = np.array(x)

        uniq_dim = len(self.geneconv_list[0].x) - len(self.shared_parameters)
     #   print(len(self.geneconv_list))
     #   print(uniq_dim)
        shared_x = self.x[len(self.geneconv_list) * uniq_dim:]
        for geneconv_idx in range(len(self.geneconv_list)):
            geneconv = self.geneconv_list[geneconv_idx]
            uniq_x = self.x[geneconv_idx * uniq_dim : (geneconv_idx + 1) * uniq_dim]
            geneconv.update_by_x(self.combine_x(uniq_x, shared_x))
     #       print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
     #       print(self.geneconv_list[1].nsites)
     #       print(self.geneconv_list[1].x)

    def save_x(self):

        save = self.x
        if self.ifmodel=="old":
           save_file = self.save_name
        else:
            save_file = self.save_name1
        print(save_file)
        np.savetxt(save_file, save.T)

    def initialize_by_save(self, save_file):
        self.x = np.loadtxt(open(save_file, 'r'))
        self.update_by_x(self.x)


    def objective_wo_gradient(self,x):

        self.x[-2] = x[0]
        self.x[-1] = x[1]

        self.update_by_x(self.x)

        f=0

        if self.ifexp==True:

            for i in self.multiprocess_combined_list:
               f=self.geneconv_list[i]._loglikelihood3()+f

        else:
            for i in self.multiprocess_combined_list:
               f=self.geneconv_list[i]._loglikelihood2()[0]+f

        return -f



    def get_Hessian(self):

        if self.ifexp==True:
             H = nd.Hessian(self.objective_wo_gradient)(np.float128([np.exp(self.x[-2]),self.x[-1]]))
        else:
            H = nd.Hessian(self.objective_wo_gradient)(np.float128([(self.x[-2]), self.x[-1]]))


        H=np.linalg.inv(H)

        return H


    def pool_get_summary(self, num_jsgeneconv):

        result=self.geneconv_list[num_jsgeneconv].get_summary()
        return result



    def get_summary(self):

        self.update_by_x(self.x)

        dd = deepcopy(self.x)
        self.update_by_x(dd)

        with Pool(processes=len(self.geneconv_list)) as pool:
               results = pool.map(self.pool_get_summary, range(len(self.geneconv_list)))

        igc_tot= np.sum([result[0] for result in results])
        mutation_tot=np.sum([result[1] for result in results])


        return igc_tot,mutation_tot






### for fix ind or shared
    def _process_objective_and_gradient_fix(self, num_jsgeneconv,output):
        if self.Model=="MG94":

                self.geneconv_list[num_jsgeneconv].Force=self.Force_share
                tauini=deepcopy(self.fixtau[num_jsgeneconv])

                if self.ifmodel == "old":
                      omegaini = deepcopy(self.fixomega[num_jsgeneconv])
                      self.geneconv_list[num_jsgeneconv].get_mle(display=False,tauini=tauini,omegaini=omegaini,ifseq=True)
                      self.fixtau[num_jsgeneconv]=deepcopy(self.geneconv_list[num_jsgeneconv].tau)
                      self.fixomega[num_jsgeneconv] = deepcopy(self.geneconv_list[num_jsgeneconv].omega)
                else:
                    kini=deepcopy(self.fixk[num_jsgeneconv])
                    self.geneconv_list[num_jsgeneconv].get_mle(display=False, tauini=tauini,kini=kini, ifseq=True)
                    self.fixtau[num_jsgeneconv] = deepcopy(self.geneconv_list[num_jsgeneconv].tau)

                self.geneconv_list[num_jsgeneconv].Force = None
                result1 = self.geneconv_list[num_jsgeneconv].objective_and_gradient(False,
                                                                                   self.geneconv_list[num_jsgeneconv].x)
                result = [result1, self.geneconv_list[num_jsgeneconv].x, num_jsgeneconv]
                output.put(result)

        else:


              #  self.update_by_x(self.x)


                self.geneconv_list[num_jsgeneconv].Force = self.Force_share
                tauini = deepcopy(self.fixtau[num_jsgeneconv])
                if self.ifmodel=="old":
                    self.geneconv_list[num_jsgeneconv].get_mle(display=False,tauini=tauini,ifseq=True)


                else:
                 #   print(num_jsgeneconv)
                    kini=deepcopy(self.fixk[num_jsgeneconv])
                    #if num_jsgeneconv==1:
                    #    print(self.geneconv_list[num_jsgeneconv].nsites)
                    #    print(self.geneconv_list[num_jsgeneconv].x)
                    #    self.geneconv_list[num_jsgeneconv].get_mle(display=True, tauini=tauini,kini=kini, ifseq=True)
                   # else:


                    self.geneconv_list[num_jsgeneconv].get_mle(display=False, tauini=tauini, kini=kini, ifseq=True)


                self.fixtau[num_jsgeneconv] = deepcopy(self.geneconv_list[num_jsgeneconv].tau)
                self.geneconv_list[num_jsgeneconv].Force = None
                result1 = self.geneconv_list[num_jsgeneconv].objective_and_gradient(False,
                                                                                   self.geneconv_list[num_jsgeneconv].x)
                result=[result1, self.geneconv_list[num_jsgeneconv].x,num_jsgeneconv]

                output.put(result)


    def reorder(self,list):
        listnew=[]
        for i in self.multiprocess_combined_list:
            for j in self.multiprocess_combined_list:
           #     print(i)
          #      print(j)
           #     print(list[j])
          #      print(list[j][2])

                if self.siteslist[i]==list[j][2]:
                       listnew.append(list[j])

        return listnew

# new parall

    def pool_process_objective_and_gradient_fix(self, num_jsgeneconv):
        if self.Model == "MG94":

            self.geneconv_list[num_jsgeneconv].Force = self.Force_share
            tauini = deepcopy(self.fixtau[num_jsgeneconv])

            if self.ifmodel == "old":
                omegaini = deepcopy(self.fixomega[num_jsgeneconv])
                self.geneconv_list[num_jsgeneconv].get_mle(display=False, tauini=tauini, omegaini=omegaini, ifseq=True)
                self.fixtau[num_jsgeneconv] = deepcopy(self.geneconv_list[num_jsgeneconv].tau)
                self.fixomega[num_jsgeneconv] = deepcopy(self.geneconv_list[num_jsgeneconv].omega)
            else:
                kini = deepcopy(self.fixk[num_jsgeneconv])
                self.geneconv_list[num_jsgeneconv].get_mle(display=False, tauini=tauini, kini=kini, ifseq=True)
                self.fixtau[num_jsgeneconv] = deepcopy(self.geneconv_list[num_jsgeneconv].tau)

            self.geneconv_list[num_jsgeneconv].Force = None
            result1 = self.geneconv_list[num_jsgeneconv].objective_and_gradient(False,
                                                                                self.geneconv_list[num_jsgeneconv].x)
            result = [result1, self.geneconv_list[num_jsgeneconv].x, num_jsgeneconv]
            return result

        else:


            self.geneconv_list[num_jsgeneconv].Force = self.Force_share
            tauini = deepcopy(self.fixtau[num_jsgeneconv])
            if self.ifmodel == "old":
                self.geneconv_list[num_jsgeneconv].get_mle(display=False, tauini=tauini, ifseq=True)


            else:
                #   print(num_jsgeneconv)
                kini = deepcopy(self.fixk[num_jsgeneconv])
                # if num_jsgeneconv==1:
                #    print(self.geneconv_list[num_jsgeneconv].nsites)
                #    print(self.geneconv_list[num_jsgeneconv].x)
                #    self.geneconv_list[num_jsgeneconv].get_mle(display=True, tauini=tauini,kini=kini, ifseq=True)
                # else:

                self.geneconv_list[num_jsgeneconv].get_mle(display=False, tauini=tauini, kini=kini, ifseq=True)

            self.fixtau[num_jsgeneconv] = deepcopy(self.geneconv_list[num_jsgeneconv].tau)
            self.geneconv_list[num_jsgeneconv].Force = None
            result1 = self.geneconv_list[num_jsgeneconv].objective_and_gradient(False,
                                                                                self.geneconv_list[num_jsgeneconv].x)
            result = [result1, self.geneconv_list[num_jsgeneconv].x, num_jsgeneconv]

            return result


    def pool_obj(self,x):


        print(x)

        if self.ifmodel!="old":

                if len(self.shared_parameters_for_k)==1:
                        self.x[-1] = deepcopy(x)
                        self.fixk = np.ones(len(self.paralog_list)) * (x)
                else:
                        self.x[-1] = deepcopy(x[1])
                        self.x[-2] = deepcopy(x[0])
                        self.fixk = np.ones(len(self.paralog_list)) * (x[1])
                        self.fixtau = np.ones(len(self.paralog_list)) * np.exp(x[0])


        else:

            if self.Model == "HKY":
                for i in self.multiprocess_combined_list:
                    self.fixtau[i] = np.exp(x)
                    self.x[-1] = deepcopy(x)
            if self.Model == "MG94":
                if len(self.shared_parameters) == 1:
                    if self.shared_parameters == 4:
                        self.x[-1] = deepcopy(x)
                        for i in self.multiprocess_combined_list:
                            self.fixomega[i] = np.exp(x)
                    else:
                        self.x[-1] = deepcopy(x)
                        for i in self.multiprocess_combined_list:
                            self.fixtau[i] = np.exp(x)
                else:
                    self.x[-1] = deepcopy(x[1])
                    self.x[-2] = deepcopy(x[0])
                    for i in self.multiprocess_combined_list:
                        self.fixomega[i] = np.exp(x[0])
                        self.fixtau[i] = np.exp(x[1])

        dd=deepcopy(self.x)
        self.update_by_x(dd)

        with Pool(processes=len(self.geneconv_list)) as pool:
               results = pool.map(self.pool_process_objective_and_gradient_fix, range(len(self.geneconv_list)))

        f = np.sum([result[0][0] for result in results]) / len(self.paralog_list)
        # uniq_derivatives will get unique derivatives for each gene
        # for  shared parameter, the derivatives is computed as sum of all genes' corresponding derivaties
        shared_derivatives = [[result[0][1][idx] for idx in range(len(result[0][1])) if idx in self.shared_parameters]
                              for
                              result in results]

        g = np.sum(shared_derivatives, axis=0) / len(self.paralog_list)

        uniq_para = np.concatenate(
            [[result[1][idx] for idx in range(len(result[1])) if not idx in self.shared_parameters] for result in
             results])
        shared_para = [results[0][1][idx] for idx in range(len(self.geneconv_list[0].x)) if
                       idx in self.shared_parameters]
        self.x = deepcopy(np.concatenate((uniq_para, shared_para)))

        print('log  likelihhood = ', f)
        print('Gradient = ', g)
        if self.ifmodel == "old":
            print('x = ', self.x)
        else:
            print('non-share x = ', self.x[:-1])
            print('K = ', (self.x[-1]))

        # Now save parameter values
        if self.ifmodel == "old":
            self.auto_save += 1
            if self.auto_save == JointAnalysis_nest.auto_save_step:
                self.save_x()
                self.auto_save = 0

        else:
            self.auto_save1 += 1
            if self.auto_save1 == JointAnalysis_nest.auto_save_step:
                self.save_x()
                self.auto_save1 = 0

        return f, g


    def obj(self, x):

        print(x)

        if self.ifmodel!="old":

                if len(self.shared_parameters_for_k)==1:
                        self.x[-1] = deepcopy(x)
                        self.fixk = np.ones(len(self.paralog_list)) * (x)
                else:
                        self.x[-1] = deepcopy(x[1])
                        self.x[-2] = deepcopy(x[0])
                        self.fixk = np.ones(len(self.paralog_list)) * (x[1])
                        self.fixtau = np.ones(len(self.paralog_list)) * np.exp(x[0])


        else:

            if self.Model == "HKY":
                for i in self.multiprocess_combined_list:
                    self.fixtau[i] = np.exp(x)
                    self.x[-1] = deepcopy(x)
            if self.Model == "MG94":
                if len(self.shared_parameters) == 1:
                    if self.shared_parameters == 4:
                        self.x[-1] = deepcopy(x)
                        for i in self.multiprocess_combined_list:
                            self.fixomega[i] = np.exp(x)
                    else:
                        self.x[-1] = deepcopy(x)
                        for i in self.multiprocess_combined_list:
                            self.fixtau[i] = np.exp(x)
                else:
                    self.x[-1] = deepcopy(x[1])
                    self.x[-2] = deepcopy(x[0])
                    for i in self.multiprocess_combined_list:
                        self.fixomega[i] = np.exp(x[0])
                        self.fixtau[i] = np.exp(x[1])

        dd=deepcopy(self.x)
        self.update_by_x(dd)


        output = mp.Queue()


        # Setup a list of processes that we want to run
        processes = [
            mp.Process(target=self._process_objective_and_gradient_fix, args=(i,  output)) \
            for i in self.multiprocess_combined_list]



        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        results = [output.get() for p in processes]

        results=self.reorder(results)

     #   print(results)

        f = np.sum([result[0][0] for result in results])/len(self.paralog_list)
        # uniq_derivatives will get unique derivatives for each gene
        # for  shared parameter, the derivatives is computed as sum of all genes' corresponding derivaties
        shared_derivatives = [[result[0][1][idx] for idx in range(len(result[0][1])) if idx in self.shared_parameters] for
                              result in results]

        g =  np.sum(shared_derivatives, axis=0)/len(self.paralog_list)


        uniq_para=np.concatenate([[result[1][idx] for idx in range(len(result[1])) if not idx in self.shared_parameters] for result in results])
        shared_para=[results[0][1][idx] for idx in range(len(self.geneconv_list[0].x)) if idx in self.shared_parameters]
        self.x = deepcopy(np.concatenate((uniq_para, shared_para)))



        print('log  likelihhood = ', f)
        print('Gradient = ',g)
        if self.ifmodel=="old":
             print('x = ', self.x)
        else:
            print('non-share x = ', self.x[:-1])
            print('K = ',(self.x[-1]))

        # Now save parameter values
        if self.ifmodel == "old":
            self.auto_save += 1
            if self.auto_save == JointAnalysis_nest.auto_save_step:
                self.save_x()
                self.auto_save = 0

        else:
            self.auto_save1 += 1
            if self.auto_save1 == JointAnalysis_nest.auto_save_step:
                self.save_x()
                self.auto_save1 = 0


        return f, g




    def get_nest_mle(self,opt="LBFGS"):


        self.update_by_x(deepcopy(self.x))


        if self.ifmodel=="old":

            if self.Model=="HKY":
                        guess_x=self.x[-1]
            if self.Model=="MG94":
                if len(self.shared_parameters)==1:
                    if self.shared_parameters==4:
                        guess_x = self.x[-1]
                    else:

                        guess_x = self.x[-1]
                else:
                        guess_x=np.zeros(2)
                        guess_x[0] = self.x[-2]
                        guess_x[1] = self.x[-1]

        else:
                if len(self.shared_parameters_for_k) == 1:
                        guess_x =  deepcopy(self.x[-1])
                else:
                    guess_x = np.zeros(2)
                    guess_x[0] = deepcopy(self.x[-2])
                    guess_x[1] = deepcopy(self.x[-1])

        if self.Model=="HKY":
            nmax=100
        else:
            nmax=40

        if opt=="LBFGS":
            result = scipy.optimize.minimize(self.pool_obj, guess_x, jac=True, method='L-BFGS-B')
        elif opt=="BFGS":
                result = scipy.optimize.minimize(self.pool_obj, guess_x, jac=True, method='BFGS')

        else:
            bnds = [(-5.0, 4.0)] * 1
            bnds.extend([(-5.0, 50.0)] * (1))
            result = scipy.optimize.basinhopping(self.pool_obj, guess_x,
                                             minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True,'bounds':bnds},
                                             niter=nmax)


        self.save_x()
      #  print(self.x)
        self.update_by_x(self.x)
        print(result)


        return result


    def em_joint(self,epis=0.01,MAX=2,opt="LBFGS"):
        ll0=self.get_nest_mle()["fun"]
        old_sum=self.get_summary()
        psK =deepcopy(self.geneconv_list[0].K)
        self.oldtau=deepcopy(self.geneconv_list[0].tau)

        self.ifmodel = "EM_full"
        self.initialize_x()
        print(" We assume K is shared and tau may be shared:")

   #     print(self.x)

        ll1=self.get_nest_mle(opt=opt)["fun"]
        K = deepcopy(self.geneconv_list[0].K)
        difference = abs(K - psK)


        print("EMcycle:")
        print(0)
        print(self.geneconv_list[0].K)
        print(self.geneconv_list[0].tau)
        print(ll1)
        print("xxxxxxxxxxxxxxxxx")
        print("xxxxxxxxxxxxxxxxx")

        print("\n")


        i=1
        while i<=MAX and difference >=epis:
            psK = deepcopy(K)
            for ii in range(len(self.paralog_list)):
                 self.geneconv_list[ii].id = self.geneconv_list[ii].compute_paralog_id()
            ll1 = self.get_nest_mle(opt=opt)["fun"]
            K = deepcopy(self.geneconv_list[0].K)
            difference = abs(K - psK)


            print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("EMcycle:")
            print(i)
            i = i + 1
            print(self.geneconv_list[0].K)
            print(self.geneconv_list[0].tau)
            print(ll1)
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")

            print("\n")

        new_sum = self.get_summary()

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("old tau:")
        print(self.oldtau)
        print("old ll:")
        print(ll0)
        print("old sum:")
        print(old_sum)
        print("new sum:")
        print(new_sum)
        print(" K:")
        print(self.geneconv_list[0].K)
        print("new tau:")
        print(self.geneconv_list[0].tau)
        print("new ll:")
        print(ll1)
        print("old igc prop", old_sum[0] / old_sum[1])
        print("new igc prop", new_sum[0] / new_sum[1])
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")


        for i in range(len(self.paralog_list)):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(i)
            print(self.geneconv_list[i].id)
            print(self.geneconv_list[i].x)




        print(self.get_Hessian())
        self.ifexp=True
        print(self.get_Hessian())


        return ll1













if __name__ == '__main__':
    paralog_1 = ['YLR406C', 'YDL075W']
    paralog_2 = ['YDR418W', 'YEL054C']
    Force = None
    alignment_file_1 = '../test/YLR406C_YDL075W_test_input.fasta'
    alignment_file_2 = '../test/YDR418W_YEL054C_input.fasta'
    newicktree = '../test/YeastTree.newick'

    paralog_list = [paralog_1, paralog_2]
    IGC_Omega = None
    Shared = [4]
    alignment_file_list = [alignment_file_1, alignment_file_2]
    Model = 'HKY'

    joint_analysis = JointAnalysis_nest(alignment_file_list,  newicktree, paralog_list, Shared = Shared,
                                   IGC_Omega = None, Model = Model, Force = Force,Force_share={4:0},
                                   shared_parameters_for_k=[4,5],Force_share_k={4:0,5:0},tauini=6.0,kini=1.1,
                                   save_path = '../test/save/')



    joint_analysis.em_joint()
  #  print([joint_analysis.x[-2],joint_analysis.x[-1]])



  #  joint_analysis.geneconv_list[0].Force = joint_analysis.Force_share
   # tauini = joint_analysis.fixtau[0]
  #  result = joint_analysis.geneconv_list[0].get_mle(tauini=tauini, ifseq=True)

 #   print(joint_analysis.geneconv_list[0].Force)



   # print(joint_analysis.objective_and_gradient_multi_threaded(joint_analysis.x))
    # print(joint_analysis.objective_and_gradient(joint_analysis.x))
    # joint_analysis.get_mle()
    # joint_analysis.get_summary('../test/save/test_summary.txt')
