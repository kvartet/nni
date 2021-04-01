import copy
import logging

import numpy as np
from schema import Optional, Schema
from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import NodeType, OptimizeMode, extract_scalar_reward, split_index

_logger = logging.getLogger(__name__)

class TPETuner(Tuner):
    def __init__(self,):
        self.n_startup_jobs = 20
        self.prior_weight = 1.0
        self.oloss_gamma = 0.25
        self.lf = 25
        self.n_ei_candidates = 24
        self.eps = 1e-12

        # public static RandomNumberGenerator rng = new RandomNumberGenerator();

        # private SearchSpace space;
        # private bool minimize;
        # private Dictionary<int, TpeParameters> parameters = new Dictionary<int, TpeParameters>();
        # private HashSet<int> running = new HashSet<int>();
        # private List<Result> history = new List<Result>();
        # private double lie = Double.PositiveInfinity;


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        pass


    def generate_parameters(self, parameter_id, **kwargs):
        pass

    def update_search_space(self, search_space):
        pass




