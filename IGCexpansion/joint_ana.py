#! /usr/bin/python3
# coding=utf-8

from IGCexpansion.CodonGeneconv import *
from IGCexpansion.em_pt1 import *
import numdifftools as nd
from multiprocessing import Pool


class JointAnalysis:
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
                 shared_parameters_for_k=[5,6],
                 inibranch=0.1,
                 kini=1.1,
                 tauini=0.4,
                 ifDNA=False):
        # first check if the length of all input lists agree
        assert (len(set([len(alignment_file_list), len(paralog_list)])) == 1)
        # doesn't allow IGC-specific omega for HKY model
        assert(Model == 'MG94' or IGC_Omega is None)

        self.save_path     = save_path
        self.Model         = Model
        self.IGC_Omega     = IGC_Omega
        self.paralog_list  = paralog_list
        self.x             = None
        self.multiprocess_combined_list = multiprocess_combined_list
        self.ifDNA =ifDNA


        #share k version:
        self.ifmodel = "old"

        if Shared is None:
            self.shared_parameters = []
        else:
            self.shared_parameters = Shared

        grand_save_name, individual_save_names = self.get_save_file_names(save_name)
        self.geneconv_list = [Embrachtau1(tree_newick = tree_newick, alignment = alignment_file_list[i], paralog = paralog_list[i],
                                              Model = Model,  nnsites = nnsites,
                                              clock = False, Force = Force, save_path = save_path, save_name = individual_save_names[i],
                                              post_dup = post_dup,ifmodel="old",inibranch=inibranch,kini=kini,tauini=tauini,ifDNA=self.ifDNA)
                              for i in range(len(alignment_file_list))]
        self.save_name     = grand_save_name

        self.auto_save = 0
        self.auto_save1 = 0
        self.initialize_x()
        self.shared_parameters_for_k=shared_parameters_for_k


    def initialize_x(self):
        if self.ifmodel == "old":
            if os.path.isfile(self.save_name):
                self.initialize_by_save(self.save_name)
                print('Successfully loaded parameter value from ' + self.save_name)
            else:

                   single_x = self.geneconv_list[0].x
                   shared_x = [single_x[i] for i in self.shared_parameters]
                   print(shared_x)

                   unique_x = [single_x[i] for i in range(len(single_x)) if not i in self.shared_parameters] * len(
                       self.geneconv_list)
                   self.unique_len=len(unique_x)
                   self.x = np.array(unique_x + shared_x)

        else:
            self.save_name1 = self.get_save_file_names(None)[0]
            self.shared_parameters = self.shared_parameters_for_k

            if os.path.isfile(self.save_name1):
                for i in range(len(self.paralog_list)):
                       self.geneconv_list[i].renew_em_joint()
                self.initialize_by_save(self.save_name1)
                # idea of EM
                self.update_by_x(self.x)
                for ii in range(len(self.paralog_list)):
                    self.geneconv_list[ii].id = self.geneconv_list[ii].compute_paralog_id()
                print(self.geneconv_list[0].id)
                self.update_by_x(self.x)
                for ii in range(len(self.paralog_list)):
                    self.geneconv_list[ii].id = self.geneconv_list[ii].compute_paralog_id()
                print(self.geneconv_list[0].id)
                self.update_by_x(self.x)
                for ii in range(len(self.paralog_list)):
                    self.geneconv_list[ii].id = self.geneconv_list[ii].compute_paralog_id()
                print(self.geneconv_list[0].id)
                print(self.geneconv_list[0].K)

                print('Successfully loaded parameter value from ' + self.save_name1)


            else:
                for i in range(len(self.paralog_list)):
                       self.geneconv_list[i].renew_em_joint()

                single_x = self.geneconv_list[0].x
                shared_x = [single_x[i] for i in self.shared_parameters]
                unique_x = np.concatenate([[self.geneconv_list[ii].x[idx] for idx in range(len(self.geneconv_list[0].x)) if
                                 not idx in self.shared_parameters]
                                for ii in self.multiprocess_combined_list])
                self.unique_len = len(unique_x)
                self.x = np.concatenate((unique_x, shared_x))

        self.update_by_x(self.x)

        if self.multiprocess_combined_list is None:
            self.multiprocess_combined_list = range(len(self.geneconv_list))

    def get_save_file_names(self, save_name):
        if len(self.shared_parameters):
            model_string = self.Model + '_withSharing'
        else:
            model_string = self.Model

        if save_name is None:
            if self.IGC_Omega is None:
                general_save_name = self.save_path + 'Joint_' + model_string + '_' + str(len(self.paralog_list)) + '_pairs_grand_save.txt'
            else:
                general_save_name = self.save_path + 'Joint_' + model_string + '_twoOmega_' + str(len(self.paralog_list)) + '_pairs_grand_save.txt'
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
        shared_x = self.x[len(self.geneconv_list) * uniq_dim:]
        for geneconv_idx in range(len(self.geneconv_list)):
            geneconv = self.geneconv_list[geneconv_idx]
            uniq_x = self.x[geneconv_idx * uniq_dim : (geneconv_idx + 1) * uniq_dim]
            geneconv.update_by_x(self.combine_x(uniq_x, shared_x))

    def get_original_bounds(self):

        if self.ifmodel != "old":
            bnds = [(None, -0.05)] * 3
            bnds.extend([(None, 6.0)] * (2))
            if self.Model=="MG94":
                bnds.extend([(None, 6.0)] * (1))
                bnds.extend([(-20.0, 40.0)] * (1))
                bnds.extend([(-10.0, 4.0)]*(len(self.geneconv_list[0].x) - 7))

            else:
                bnds.extend([(-20.0, 40.0)] * (1))
                bnds.extend([(-10.0, 4.0)]*(len(self.geneconv_list[0].x) - 6))
        else:
            bnds = [(None, -0.05)] * 3
            bnds.extend([(None, 6.0)] * (2))
            if self.Model=="MG94":
                bnds.extend([(None, 6.0)] * (1))
                bnds.extend([(-10.0, 4.0)]*(len(self.geneconv_list[0].x) - 6))

            else:
                bnds.extend([(-10.0, 4.0)] * (len(self.geneconv_list[0].x) - 5))


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

    def objective_wo_gradient(self, x):
        f=0
        if  len(self.shared_parameters_for_k)==2:
            if len(self.shared_parameters_for_k) == 1:
                self.x[-1] = x[0]
            else:
                self.x[-2] = x[0]
                self.x[-1] = x[1]
            self.update_by_x(self.x)

            for i in self.multiprocess_combined_list:
                self.geneconv_list[i].update_by_x(self.geneconv_list[i].x)
                self.geneconv_list[i].id = self.geneconv_list[i].compute_paralog_id()
                self.geneconv_list[i].update_by_x(self.geneconv_list[i].x)
                f = self.geneconv_list[i]._loglikelihood2()[0] + f

        else:
            if len(self.shared_parameters_for_k) == 1:
                self.x[-1] = x[0]
            else:
                self.x[-2] = (x[0])
                self.x[-1] = x[1]
            self.update_by_x(self.x)

            for i in self.multiprocess_combined_list:
                self.geneconv_list[i].update_by_x(self.geneconv_list[i].x)
                self.geneconv_list[i].id = self.geneconv_list[i].compute_paralog_id()
                self.geneconv_list[i].update_by_x(self.geneconv_list[i].x)
                f = self.geneconv_list[i]._loglikelihood2()[0] + f

        return -f


    def _pool_objective_and_gradient(self, num_jsgeneconv):
        result = self.geneconv_list[num_jsgeneconv].objective_and_gradient(False, self.geneconv_list[num_jsgeneconv].x)
        return result


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
            if num_jsgeneconv==1:
                print(self.geneconv_list[1].nsites)
                print(result[0])


            output.put(result)


    def objective_and_gradient_multi_threaded(self, x):

        self.update_by_x(x)

        with Pool(processes=len(self.geneconv_list)) as pool:
            results = pool.map(self._pool_objective_and_gradient, range(len(self.geneconv_list)))



        #  results store lists, each list store gradients for parameters in corresponding gene
        ##        pool = mp.Pool(processes = self.num_processes)
        ##        results = [pool.apply(psjsgeneconv.objective_and_gradient, args = (display, x))\
        ##                   for psjsgeneconv in self.psjsgeneconv_list]

        f = sum([result[0] for result in results])
        # uniq_derivatives will get unique derivatives for each gene
        uniq_derivatives = np.concatenate([[result[1][idx] for idx in range(len(result[1])) if not idx in self.shared_parameters] for result in results])
        # for  shared parameter, the derivatives is computed as sum of all genes' corresponding derivaties
        shared_derivatives = [[result[1][idx] for idx in range(len(result[1])) if idx in self.shared_parameters] for result in results]
        g = np.concatenate((uniq_derivatives, np.sum(shared_derivatives, axis = 0)))

        print('log  likelihhood = ', f)
        print('Current x array = ', self.x)
        print('exp x = ', np.exp(self.x))
        print('Gradient = ', g)

        # Now save parameter values
        if self.ifmodel=="old":
            self.auto_save += 1
            if self.auto_save == JointAnalysis.auto_save_step:
                self.save_x()
                self.auto_save = 0

        else:
            self.auto_save1 += 1
            if self.auto_save1 == JointAnalysis.auto_save_step:
                self.save_x()
                self.auto_save1 = 0

        return f, g

    def get_mle(self, parallel = True):
        self.update_by_x(self.x)

        guess_x = self.x

        if parallel:
            result = scipy.optimize.minimize(self.objective_and_gradient_multi_threaded, guess_x, jac=True, method='L-BFGS-B', bounds=self.combine_bounds(),
                                             )
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


    def get_summary(self, summary_file):
        individual_results = [self.geneconv_list[i].get_summary(True) for i in range(len(self.paralog_list))]
        summary = np.array([res[0] for res in individual_results])
        label = individual_results[0][1]

        footer = ' '.join(label)  # row labels
        header = ' '.join(['_'.join(paralog) for paralog in self.paralog_list])
        np.savetxt(summary_file, summary.T, delimiter = ' ', header = header, footer = footer)



    def get_Hessian(self):


            if len(self.shared_parameters_for_k) == 2:
          #      basic =  np.maximum(np.exp(self.x[-2]) / (2 * (np.maximum(np.log(np.exp(self.x[-2]) + 1), 1))) - 0.5,0.2)
           #     step = nd.step_generators.MaxStepGenerator(base_step=basic)
             #   H = nd.Hessian(self.objective_wo_gradient, step = step)(np.float128([np.exp(self.x[-2]), self.x[-1]]))

                   H = nd.Hessian(self.objective_wo_gradient)(np.float128([(self.x[-2]), self.x[-1]]))
            else:
                H = nd.Hessian(self.objective_wo_gradient)(np.float128([self.x[-1]]))



            return H


    def em_joint(self,epis=0.1,MAX=3):
        ll0=self.get_mle()["fun"]
      #  print(ll0)
        if len(self.shared_parameters_for_k) == 1:
            self.oldtau=deepcopy(([self.geneconv_list[i].tau for i in range(len(self.paralog_list))]))
        else:
            self.oldtau=deepcopy(self.geneconv_list[1].tau)

        pstau =deepcopy(([self.geneconv_list[i].tau for i in range(len(self.paralog_list))]))
        pstau=np.sum(pstau)
        self.ifmodel = "EM_full"
        self.initialize_x()

        self.get_mle()
        tau = deepcopy(([self.geneconv_list[i].tau for i in range(len(self.paralog_list))]))
        tau=np.sum(tau)
        difference = abs(tau - pstau)

        print("EMcycle:")
        print(0)
        print(self.geneconv_list[1].K)
        if len(self.shared_parameters_for_k)==1:
            for i in range(len(self.geneconv_list)):
                print(self.geneconv_list[i].tau)
                print(self.geneconv_list[i].omega)
                print(self.geneconv_list[i].ll)
        else:
            print(self.geneconv_list[1].tau)
        print("xxxxxxxxxxxxxxxxx")
        print("xxxxxxxxxxxxxxxxx")
        print("xxxxxxxxxxxxxxxxx")
        print("\n")


        i=1
        while i<=MAX and difference >=epis:
            pstau = deepcopy(tau)
            for ii in range(len(self.paralog_list)):
                 self.geneconv_list[ii].id = self.geneconv_list[ii].compute_paralog_id()
            ll1=self.get_mle()["fun"]
            tau = deepcopy(np.exp([self.geneconv_list[i].x[5] for i in range(len(self.paralog_list))]))
            tau = np.sum(tau)
            difference = abs(tau - pstau)

            print("EMcycle:")
            print(i)
            i = i + 1

            print(self.geneconv_list[1].K)
            if len(self.shared_parameters_for_k) == 1:
                for i in range(len(self.geneconv_list)):
                    print(self.geneconv_list[i].tau)
                    print(self.geneconv_list[i].omega)
                    print(self.geneconv_list[i].ll)
            else:
                print(self.geneconv_list[1].tau)
            print("xxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxx")
            print("\n")

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("old tau:")
        print(self.oldtau)
        print("old ll:")
        print(ll0)
        print("new ll:")
        print(ll1)

        print(" K:")
        print(self.geneconv_list[0].K)
        print("new tau:")
        if len(self.shared_parameters_for_k) == 1:
            for i in range(len(self.geneconv_list)):
                print(self.geneconv_list[i].tau)
        else:
            print(self.geneconv_list[1].tau)


        for i in range(len(self.paralog_list)):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(i)
            print(self.geneconv_list[i].id)
            print(self.geneconv_list[i].x)

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(self.get_Hessian())

#        print(self.get_Hessian())











if __name__ == '__main__':
    paralog_1 = ['YLR406C', 'YDL075W']
    paralog_2 = ['YDR418W', 'YEL054C']
    Force = None
    alignment_file_1 = '../test/YLR406C_YDL075W_test_input.fasta'
    alignment_file_2 = '../test/YDR418W_YEL054C_input.fasta'
    newicktree = '../test/YeastTree.newick'

    paralog_list = [paralog_1, paralog_2]
    IGC_Omega = None
    Shared = [5]
    alignment_file_list = [alignment_file_1, alignment_file_2]
    Model = 'HKY'

    joint_analysis = JointAnalysis(alignment_file_list,  newicktree, paralog_list, Shared = [],
                                   IGC_Omega = None, Model = Model, Force = Force,
                                   save_path = '../test/save/',shared_parameters_for_k=[5])

    joint_analysis.em_joint()


    #joint_analysis.get_mle()



   # print(joint_analysis.objective_and_gradient_multi_threaded(joint_analysis.x))
    # print(joint_analysis.objective_and_gradient(joint_analysis.x))
    # joint_analysis.get_mle()
    # joint_analysis.get_summary('../test/save/test_summary.txt')
