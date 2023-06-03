
from __future__ import print_function


import os



#pip uninstall IGCexpansion
#test9
#sbatch -c 4 --mem-per-cpu=16G RunJoint_Yeast.py


def build_dic(dd=3):
    list = []
    for i in range(14):
        list.append("\n"+"cd")
      #  p="\n"+"cd final1/simulation/yeast_YER102W_YBL072C/test\ copy\ "+str(i)+"/"
      #  p = "\n" + "cd final1/simulation/pillar2158/test\ copy\ " + str(i) + "/"
    #    p="\n" + "cd final1/simub_y13_DNA/test1\ copy\ " + str(i) + "/"
    #    p = "\n" + "cd final1/simulation/testfile/testgnew/test" + str(i) + "/"
        p = "\n" + "cd final2/yeast_unlimit_tau_DNA2/test" + str(i) + "/"
     #   p="\n" + "cd final1/YeastSeq4r2" + str(i)
        list.append(p)
      #  list.append("\n" + "sbatch -c 2 --mem-per-cpu=16G RunJoint_Yeast.py")
        list.append("\n"+"sbatch  Run_IS_IGC_Old.py")

  #  with open('../save/read.txt') as f:
    #    lines = f.readlines()
    with open('../save/read.txt', 'wb') as f:
        for file in list:
            f.write(file.encode('utf-8'))


def read_dic(dd=3):
        folder = '/Users/tanchumin/Desktop/fish_unlimit_tau_DNA2/'
        sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

        list = []
        for i in range(len(sub_folders)):
            list.append("\n" + "cd")
            p = "\n" + "cd final2/fish_unlimit_tau_DNA2/" + str(sub_folders[i]) + "/"
            list.append(p)
            list.append("\n" + "sbatch  Run_IS_IGC_Old.py")
        with open('../save/read.txt', 'wb') as f:
            for file in list:
                f.write(file.encode('utf-8'))


#build_dic()
read_dic()