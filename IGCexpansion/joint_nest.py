#! /usr/bin/python3
# coding=utf-8

from IGCexpansion.CodonGeneconv import *
import multiprocessing as mp
from IGCexpansion.em_pt import *
import numdifftools as nd
from scipy import  *


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

        for i in range(len(self.paralog_list)):
            self.fixtau[i] = tauini
            self.fixk[i] = kini
            if self.Model == "MG94":
                  self.fixomega[i]=omegaini


    def initialize_x(self):
        if self.ifmodel == "old":
            if os.path.isfile(self.save_name):
                self.initialize_by_save(self.save_name)
                print('Successfully loaded parameter value from ' + self.save_name)
            else:

                   single_x = self.geneconv_list[0].x
                   shared_x = [single_x[i] for i in self.shared_parameters]

                   unique_x = [single_x[i] for i in range(len(single_x)) if not i in self.shared_parameters] * len(
                       self.geneconv_list)
                   self.unique_len=len(unique_x)
                   self.x = np.array(unique_x + shared_x)

        else:

            self.save_name1 = None
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


            else:
                for i in range(len(self.paralog_list)):
                       self.geneconv_list[i].renew_em_joint()

                single_x = self.geneconv_list[0].x
                shared_x = [single_x[i] for i in self.shared_parameters]


                unique_x = [single_x[i] for i in range(len(single_x)) if not i in self.shared_parameters] * len(
                    self.geneconv_list)
                self.unique_len = len(unique_x)
                self.x = np.array(unique_x + shared_x)

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

    def get_original_bounds(self):
        if self.ifmodel=="EM_full":
           tau=deepcopy(np.log(self.oldtau))


        if self.ifmodel != "old":
            bnds = [(-9.0, -0.05)] * 3
            bnds.extend([(-9.0, 6.0)] * (3))
            if self.Model=="MG94":
                bnds.extend([(-9.0, 6.0)] * (1))
                bnds.extend([(-9.0, 4.0)]*(len(self.geneconv_list[0].x) - 7))

            else:
                bnds.extend([(-9.0, 4.0)]*(len(self.geneconv_list[0].x) - 6))
        else:
            bnds = [(None, -0.05)] * 3
            bnds.extend([(None, 6.0)] * (2))
            if self.Model=="MG94":
                bnds.extend([(None, 6.0)] * (1))
                bnds.extend([(None, 4.0)]*(len(self.geneconv_list[0].x) - 6))

            else:
                bnds.extend([(None, 4.0)]*(len(self.geneconv_list[0].x) - 5))




        return bnds


    def combine_bounds(self):
        individual_bnds = self.get_original_bounds()
        combined_bounds = [individual_bnds[idx] for idx in range(len(individual_bnds)) if not idx in self.shared_parameters] * len(self.paralog_list) \
                          + [individual_bnds[idx] for idx in range(len(individual_bnds)) if idx in self.shared_parameters]
        return combined_bounds


    def objective_and_gradient(self, x):
        self.update_by_x(x)
        individual_results = [geneconv.objective_and_gradient(False, geneconv.x) for geneconv in self.geneconv_list]
        f = sum([result[0] for result in individual_results])
        uniq_derivatives = np.concatenate([[result[1][idx] for idx in range(len(result[1])) if not idx in self.shared_parameters] for result in individual_results])
        shared_derivatives = [[result[1][idx] for idx in range(len(result[1])) if idx in self.shared_parameters] for result in individual_results]
        g = np.concatenate((uniq_derivatives, np.sum(shared_derivatives, axis = 0)))
        if self.ifmodel=="old":
            self.auto_save += 1
            if self.auto_save == self.auto_save_step:
                self.save_x()
                self.auto_save = 0
        else:
            self.auto_save1 += 1
            if self.auto_save1 == self.auto_save_step:
                self.save_x()
                self.auto_save1 = 0


        print('log likelihood = ', f)
        print('Current x array = ', self.x)
        print('Derivatives = ', g)
        return f, g


    def _process_objective_and_gradient(self, num_jsgeneconv, display, x, output):
        if self.Model=="MG94":
        #    print(num_jsgeneconv,flush=True)
            self.update_by_x(x)
            result = self.geneconv_list[num_jsgeneconv].objective_and_gradient(True, self.geneconv_list[num_jsgeneconv].x)
            output.put(result)
        else:
            self.update_by_x(x)
            result = self.geneconv_list[num_jsgeneconv].objective_and_gradient(display,
                                                                               self.geneconv_list[num_jsgeneconv].x)
            output.put(result)


    def objective_and_gradient_multi_threaded(self, x):
        self.update_by_x(x)

        # Define an output queue
        output = mp.Queue()

        # Setup a list of processes that we want to run
        processes = [mp.Process(target=self._process_objective_and_gradient, args=(i, False, x, output)) \
                     for i in self.multiprocess_combined_list]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        results = [output.get() for p in processes]

        #  results store lists, each list store gradients for parameters in corresponding gene

        ##        pool = mp.Pool(processes = self.num_processes)
        ##        results = [pool.apply(psjsgeneconv.objective_and_gradient, args = (display, x))\
        ##                   for psjsgeneconv in self.psjsgeneconv_list]

        f = sum([result[0] for result in results])
        # uniq_derivatives will get unique derivatives for each gene
        uniq_derivatives = np.concatenate([[result[1][idx] for idx in range(len(result[1])) if not idx in self.shared_parameters] for result in results])
        # for  shared parameter, the derivatives is computed as sum of all genes' corresponding derivaties
        shared_derivatives = [[result[1][idx] for idx in range(len(result[1])) if idx in self.shared_parameters] for result in results]
        g = np.concatenate((uniq_derivatives, np.mean(shared_derivatives, axis = 0)))

        print('log  likelihhood = ', f)
        print('Current x array = ', self.x)
        print('exp x = ', np.exp(self.x))
        print('Gradient = ', g)

        # Now save parameter values
        if self.ifmodel=="old":
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

    def get_mle(self, parallel = True):
        self.update_by_x(self.x)

        guess_x = self.x

        if parallel:
       #     result = scipy.optimize.minimize(self.objective_and_gradient_multi_threaded, guess_x, jac=True, method='L-BFGS-B', bounds=self.combine_bounds(),
                    #                         options={ 'maxcor': 12,'ftol': 1e-8,'maxls': 30})

         #   result = scipy.optimize.minimize(self.objective_and_gradient_multi_threaded, guess_x, jac=True, method='SLSQP',
            #                         bounds=self.combine_bounds(),
            #                 options={ 'maxcor': 12,'ftol': 1e-8,'maxls': 30})


                 result=scipy.optimize.basinhopping(self.objective_and_gradient_multi_threaded, guess_x, minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True,
                                                              'bounds': self.combine_bounds()},
                                niter=150)
        else:
            result = scipy.optimize.minimize(self.objective_and_gradient, guess_x, jac=True, method='L-BFGS-B', bounds=self.combine_bounds())
        print (result)

        self.save_x()

        return result

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



    def get_Hessian(self):
        H = nd.Hessian(self.objective_wo_gradient)(np.float128((self.x[int(self.unique_len)-1:int(self.unique_len)+1])))

        return H


### for fix ind or shared
    def _process_objective_and_gradient_fix(self, num_jsgeneconv, display, output):
        if self.Model=="MG94":

                self.update_by_x(self.x)
                self.geneconv_list[num_jsgeneconv].Force=self.Force_share
                tauini=self.fixtau[num_jsgeneconv]

                if self.ifmodel == "old":
                      omegaini = self.fixomega[num_jsgeneconv]
                      self.geneconv_list[num_jsgeneconv].get_mle(display=False,tauini=tauini,omegaini=omegaini,ifseq=True)
                      self.fixtau[num_jsgeneconv]=deepcopy(self.geneconv_list[num_jsgeneconv].tau)
                      self.fixomega[num_jsgeneconv] = deepcopy(self.geneconv_list[num_jsgeneconv].omega)
                else:
                    kini=deepcopy(self.fixk[num_jsgeneconv])
                    self.geneconv_list[num_jsgeneconv].get_mle(display=False, tauini=tauini,kini=kini, ifseq=True)

                self.geneconv_list[num_jsgeneconv].Force = None
                result = self.geneconv_list[num_jsgeneconv].objective_and_gradient(True,
                                                                                   self.geneconv_list[num_jsgeneconv].x)
                result = [result, self.geneconv_list[num_jsgeneconv].x]
                output.put(result)

        else:


                self.update_by_x(self.x)

                self.geneconv_list[num_jsgeneconv].Force = self.Force_share
                tauini = deepcopy(self.fixtau[num_jsgeneconv])
                if self.ifmodel=="old":
                    self.geneconv_list[num_jsgeneconv].get_mle(display=False,tauini=tauini,ifseq=True)

                else:
                    kini=deepcopy(self.fixk[num_jsgeneconv])
                    self.geneconv_list[num_jsgeneconv].get_mle(display=False, tauini=tauini,kini=kini, ifseq=True)


                self.fixtau[num_jsgeneconv] = deepcopy(self.geneconv_list[num_jsgeneconv].tau)
                self.geneconv_list[num_jsgeneconv].Force = None
                result = self.geneconv_list[num_jsgeneconv].objective_and_gradient(False,
                                                                                   self.geneconv_list[num_jsgeneconv].x)
                result=[result, self.geneconv_list[num_jsgeneconv].x]
                output.put(result)


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
                        self.x[-1] = x
                        for i in self.multiprocess_combined_list:
                            self.fixtau[i] = np.exp(x)
                else:
                    self.x[-1] = deepcopy(x[1])
                    self.x[-2] = deepcopy(x[0])
                    for i in self.multiprocess_combined_list:
                        self.fixomega[i] = np.exp(x[0])
                        self.fixtau[i] = np.exp(x[1])



        self.update_by_x(self.x)

        output = mp.Queue()


        # Setup a list of processes that we want to run
        processes = [
            mp.Process(target=self._process_objective_and_gradient_fix, args=(i, False,  output)) \
            for i in self.multiprocess_combined_list]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        results = [output.get() for p in processes]
     #   print(results)

        f = np.sum([result[0][0] for result in results])/len(self.paralog_list)
        # uniq_derivatives will get unique derivatives for each gene
        uniq_derivatives = np.concatenate([[0 for idx in range(len(result[1])) if not idx in self.shared_parameters] for result in results])
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
             print('exp shared = ', np.exp(self.x))
        else:
            print('exp shared = ', np.exp(self.x[:-1]))
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
            result = scipy.optimize.minimize(self.obj, guess_x, jac=True, method='L-BFGS-B')
        elif opt=="LBFGS":
            if opt == "LBFGS":
                result = scipy.optimize.minimize(self.obj, guess_x, jac=True, method='BFGS')

        else:
            bnds = [(-5.0, 3.0)] * 1
            bnds.extend([(-5.0, 10.0)] * (1))
            result = scipy.optimize.basinhopping(self.obj, guess_x,
                                             minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True,'bounds':bnds},
                                             niter=nmax)


        self.save_x()
        self.update_by_x(self.x)
        print(result)


        return result


    def em_joint(self,epis=0.01,MAX=3,opt="BFGS"):
        ll0=self.get_nest_mle()["fun"]
        self.oldtau=deepcopy(self.geneconv_list[1].tau)
        psK =deepcopy(self.geneconv_list[0].K)

        self.ifmodel = "EM_full"
        self.initialize_x()
        print(" We assume K is shared and tau may be shared")

        self.get_nest_mle(opt=opt)
        K = deepcopy(self.geneconv_list[0].K)
        difference = abs(K - psK)

        print("EMcycle:")
        print(0)
        print(self.geneconv_list[0].K)
        print(self.geneconv_list[0].tau)
        print("xxxxxxxxxxxxxxxxx")
        print("xxxxxxxxxxxxxxxxx")

        print("\n")


        i=1
        while i<=MAX and difference >=epis:
            psK = deepcopy(K)
            for ii in range(len(self.paralog_list)):
                 self.geneconv_list[ii].id = self.geneconv_list[ii].compute_paralog_id()
            self.get_nest_mle(opt=opt)
            K = deepcopy(np.exp(self.geneconv_list[i].K))
            difference = abs(K - psK)


            print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("EMcycle:")
            print(i)
            i = i + 1
            print(self.geneconv_list[1].K)
            print(self.geneconv_list[1].tau)
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")

            print("\n")

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("old tau:")
        print(self.oldtau)
        print("old ll:")
        print(ll0)
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")


        for i in range(len(self.paralog_list)):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(i)
            print(self.geneconv_list[i].id)
            print(self.geneconv_list[i].x)










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


    #joint_analysis.get_mle()
  #  joint_analysis.geneconv_list[0].Force={4:0}
  #  print( joint_analysis.geneconv_list[0].Force)
  #  joint_analysis.geneconv_list[0].get_mle(tauini=55)
 #   joint_analysis.ind_ana()
    print(joint_analysis.em_joint())

  #  joint_analysis.geneconv_list[0].Force = joint_analysis.Force_share
   # tauini = joint_analysis.fixtau[0]
  #  result = joint_analysis.geneconv_list[0].get_mle(tauini=tauini, ifseq=True)

 #   print(joint_analysis.geneconv_list[0].Force)



   # print(joint_analysis.objective_and_gradient_multi_threaded(joint_analysis.x))
    # print(joint_analysis.objective_and_gradient(joint_analysis.x))
    # joint_analysis.get_mle()
    # joint_analysis.get_summary('../test/save/test_summary.txt')
