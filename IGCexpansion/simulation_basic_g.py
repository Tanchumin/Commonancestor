from __future__ import print_function
import jsonctmctree.ll, jsonctmctree.interface
from JSGeneconv import JSGeneconv
from CodonGeneconv import *
from Func import *
from copy import deepcopy
import os
from Common import *
import numpy as np
import scipy as sc

class GL:

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

