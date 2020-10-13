from IGCexpansion.CodonGeneconv import ReCodonGeneconv
import argparse, os
import numpy as np

def check_folder(folder_name):
    # if the folder doesn't exist, 
    # this function creats it by assuming the outer folder exists (potential bug...)
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

if __name__ == '__main__':
    paralog = ['EDN', 'ECP']
    Force = None
    alignment_file = './EDN_ECP_Cleaned.fasta'
    newicktree = './EDN_ECP_tree.newick'
    Force = None
    model = 'HKY'  # choose from 'HKY' and 'MG94'
    save_folder = './save/'
    check_folder(save_folder)
    save_name = save_folder + model + '_EDN_ECP_nonclock_save_change.txt'

    summary_folder = './summary/'
    check_folder(summary_folder)

    test = ReCodonGeneconv( newicktree, alignment_file, paralog, Model = model, Force = Force, clock = None, save_path = './save/', save_name = save_name,
                            tau=1, omega=0.9, kappa=0.1, inibl=0.2 )
    test.get_mle()
    test.get_ExpectedNumGeneconv()

    test.get_individual_summary(summary_path='./save/')
