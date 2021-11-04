# coding=utf-8

from IGCexpansion.CodonGeneconv import *
import multiprocessing as mp
from IGCexpansion.em_pt import *


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
                 post_dup = 'N1'):
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
                                              post_dup = post_dup,ifmodel="old")
                              for i in range(len(alignment_file_list))]
        self.save_name     = grand_save_name

        self.auto_save = 0
        self.initialize_x()

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
                   self.x = np.array(unique_x + shared_x)

        else:
                for i in range(len(self.paralog_list)):
                       self.geneconv_list[i].renew_em_joint()

                single_x = self.geneconv_list[0].x
                self.shared_parameters=[5,6]
                shared_x = [single_x[i] for i in self.shared_parameters]
                unique_x = [single_x[i] for i in range(len(single_x)) if not i in self.shared_parameters] * len(
                    self.geneconv_list)
                self.x = np.array(unique_x + shared_x)

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
        self.check_x_dim()
        self.x = np.array(x)
        uniq_dim = len(self.geneconv_list[0].x) - len(self.shared_parameters)
        shared_x = self.x[len(self.geneconv_list) * uniq_dim:]
        for geneconv_idx in range(len(self.geneconv_list)):
            geneconv = self.geneconv_list[geneconv_idx]
            uniq_x = self.x[geneconv_idx * uniq_dim : (geneconv_idx + 1) * uniq_dim]
            geneconv.update_by_x(self.combine_x(uniq_x, shared_x))

    def get_original_bounds(self):
        bnds = [(None, -0.05)] * 3
        bnds.extend([(None, None)] * (len(self.geneconv_list[0].x) - 3))
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
        self.auto_save += 1
        if self.auto_save == self.auto_save_step:
            self.save_x()
            self.auto_save = 0

        print('log likelihood = ', f)
        print('Current x array = ', self.x)
        print('Derivatives = ', g)
        return f, g

    def _process_objective_and_gradient(self, num_jsgeneconv, display, x, output):
        self.update_by_x(x)
        result = self.geneconv_list[num_jsgeneconv].objective_and_gradient(False, self.geneconv_list[num_jsgeneconv].x)
        output.put(result)

    def objective_and_gradient_multi_threaded(self, x):
        self.update_by_x(x)
        f = 0.0
        g = 0.0
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

        ##        pool = mp.Pool(processes = self.num_processes)
        ##        results = [pool.apply(psjsgeneconv.objective_and_gradient, args = (display, x))\
        ##                   for psjsgeneconv in self.psjsgeneconv_list]

        f = sum([result[0] for result in results])
        uniq_derivatives = np.concatenate([[result[1][idx] for idx in range(len(result[1])) if not idx in self.shared_parameters] for result in results])
        shared_derivatives = [[result[1][idx] for idx in range(len(result[1])) if idx in self.shared_parameters] for result in results]
        g = np.concatenate((uniq_derivatives, np.sum(shared_derivatives, axis = 0)))

        print('log likelihhood = ', f)
        print('Current x array = ', self.x)
        print('exp x = ', np.exp(self.x))
        print('Gradient = ', g)

        # Now save parameter values
        self.auto_save += 1
        if self.auto_save == JointAnalysis.auto_save_step:
            self.save_x()
            self.auto_save = 0
        return f, g

    def get_mle(self, parallel = True):
        self.update_by_x(self.x)

        guess_x = self.x

        if parallel:
            result = scipy.optimize.minimize(self.objective_and_gradient_multi_threaded, guess_x, jac=True, method='L-BFGS-B', bounds=self.combine_bounds())
        else:
            result = scipy.optimize.minimize(self.objective_and_gradient, guess_x, jac=True, method='L-BFGS-B', bounds=self.combine_bounds())
        print (result)
        self.save_x()
        return result

    def save_x(self):
        save = self.x
        save_file = self.save_name
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


    def em_joint(self,epis=0.01,MAX=5):
        self.get_mle()
        pstau = deepcopy(np.exp([self.geneconv_list[i].x[5] for i in range(len(self.paralog_list))]))
        pstau=np.sum(pstau)
        self.ifmodel = "EM_full"
        self.initialize_x()

        self.get_mle()
        tau = deepcopy(np.exp([self.geneconv_list[i].x[5] for i in range(len(self.paralog_list))]))
        tau=np.sum(tau)
        difference = abs(tau - pstau)

        print("EMcycle:")
        print(0)
        print(np.exp(self.geneconv_list[1].x[5]))
        print(np.exp(self.geneconv_list[1].x[6]))
        print("xxxxxxxxxxxxxxxxx")
        print("xxxxxxxxxxxxxxxxx")
        print("xxxxxxxxxxxxxxxxx")
        print("\n")


        i=1
        while i<=MAX and difference >=epis:
            pstau = deepcopy(tau)
            for i in range(len(self.paralog_list)):
                 self.geneconv_list[i].id = self.geneconv_list[i].compute_paralog_id()
            self.get_mle()
            i=i+1
            tau = deepcopy(np.exp([self.geneconv_list[i].x[5] for i in range(len(self.paralog_list))]))
            tau = np.sum(tau)
            difference = abs(tau - pstau)

            print("EMcycle:")
            print(i)
            print(np.exp(self.geneconv_list[1].x[5]))
            print(np.exp(self.geneconv_list[1].x[6]))
            print("xxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxx")
            print("\n")











if __name__ == '__main__':
    paralog_1 = ['YLR406C', 'YDL075W']
    paralog_2 = ['YDR418W', 'YEL054C']
    Force = None
    alignment_file_1 = '../test/YLR406C_YDL075W_test_input.fasta'
    alignment_file_2 = '../test/YDR418W_YEL054C_input.fasta'
    newicktree = '../test/YeastTree.newick'

    paralog_list = [paralog_1, paralog_2]
    IGC_Omega = None
    Shared = [6]
    alignment_file_list = [alignment_file_1, alignment_file_2]
    Model = 'MG94'

    joint_analysis = JointAnalysis(alignment_file_list,  newicktree, paralog_list, Shared = Shared,
                                   IGC_Omega = 0.8, Model = Model, Force = Force,
                                   save_path = '../test/save/')

    print(joint_analysis.geneconv_list[0].x[1])


   # print(joint_analysis.objective_and_gradient_multi_threaded(joint_analysis.x))
    # print(joint_analysis.objective_and_gradient(joint_analysis.x))
    # joint_analysis.get_mle()
    # joint_analysis.get_summary('../test/save/test_summary.txt')
