# IGCexpansion
[![Build Status](https://travis-ci.com/Tanchumin/Commonancestor.svg?branch=master)](https://travis-ci.com/xji3/IGCexpansion)

IGC paralog divergence development folder. Here, we provide different approach to detect
relationship between paralog divergence and IGC rate tau.

##### Dependent software:

[jsonctmctree package](http://jsonctmctree.readthedocs.org/en/latest/) (powerful likelihood  calculation
engine by Alex Griffing, modified by Xiang Ji)

[Biopython](http://biopython.org/wiki/Biopython)
[networkx](https://networkx.github.io/)
[numpy](https://numpy.org/)
[scipy](https://www.scipy.org/)
[numdifftools](https://pypi.org/project/numdifftools/)

(you could install them by

`
pip install --user Biopython networkx numpy scipy numdifftools
`)


#### Coding Language

Python 3.6 or higher


#### Preparation

*Mac OS / Linux*

1. To install python packages, you need to use [pip](https://pip.pypa.io/en/stable/installing/) (package management).

2. You might need to install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

*Windows*

1. I recommand using [Anaconda](https://www.anaconda.com/products/individual#windows) on Windows that ships with pip functionality and many more useful features.

2. You still need to install [git](https://git-scm.com/download/win)


#### Install python packages

1. Install jsonctmctree package by Alex Griffing (slightly modified for version updates):

`
pip install --user git+https://github.com/xji3/jsonctmctree.git
`

2. Install paralog_divergence package:

`
pip install --user git+https://github.com/Tanchumin/Commonancestor.git
`

3. Similarly install any other python packages (they should have been installed with IGCexpansion already)

`
pip install --user networkx
`

`
pip install --user Biopython
`

To uninstall:

`
pip uninstall IGCexpansion
`

##### Getting a local copy of the package

`
git clone https://github.com/Tanchumin/Commonancestor.git
`

You can now run the tutorial file or edit it to perform analyses.

`
cd IGCexpansion/tutorials/IS_IGC
`

`
python Run_IS_IGC.py
`



##### Tutorials
The latest update introduces several new features. The addition includes enhancements to various modules. 
The [em_pt1] module with class [Embrachtau1] now incorporates the number of paralog divergences, allowing for the
investigation of how the IGC  rate changes with paralog identity levels. 
The [joint_ana] module, featuring the [JointAnalysis] class, enables the analysis 
of multiple genes by utilizing a shared IGC rate and K. 
The [gls_seq] module with the [GSseq] class,
facilitates the simulation of sequences based on assigned parameters.