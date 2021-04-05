import copy
import random
import logging
import collections

import numpy as np

from schema import Optional, Schema
from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import NodeType, OptimizeMode, extract_scalar_reward, split_index

from .utils import *


_logger = logging.getLogger(__name__)


'''
tmp = {
    "branch0":{
        "_type":"choice",
        "_value":[
            {
                "_name":"tmp1",
                "batch_size":{"_type":"choice","_value":[1,2,3]},
                "kernal_size":{"_type":"choice","_value":[1,2,3]},
            },
            {
                "_name":"tmp2",
                "batch_size":{"_type":"choice","_value":[1,2,3]},
                "kernal_size":{"_type":"choice","_value":[1,2,3]},
            }
        ]
    },
    "branch1":{
        "_type":"choice",
        "_value":[
            {
                "_name":"tmp1",
                "batch_size":{"_type":"choice","_value":[1,2,3]},
                "kernal_size":{"_type":"choice","_value":[1,2,3]},
            },
            {
                "_name":"tmp2",
                "batch_size":{"_type":"choice","_value":[1,2,3]},
                "kernal_size":{"_type":"choice","_value":[1,2,3]},
            }
        ]
    }
}
'''


class ParameterRange():
    def __init__(self, name, algorithm_name, is_categorical, size,
                 categorical_values, low, high, is_log_distributed, is_integer):
        self.name = name
        # self.tag = algorithm_name + "|" + name
        self.tag = name

        self.is_categorical = is_categorical

        self.size = size
        self.categorical_values = categorical_values

        self.low = low
        self.high = high
        self.is_log_distributed = is_log_distributed
        self.is_integer = is_integer

        if is_log_distributed:
            self.high = np.log(high)
            self.low = np.log(low)

    @staticmethod
    def categorical(algorithm_name, name, values):
        return ParameterRange(name, algorithm_name, True,
                              len(values), values, np.nan, np.nan, False, False)

    @staticmethod
    def numerical(algorithm_name, name, low, high,
                  is_log_distributed=False, is_integer=False):
        return ParameterRange(name, algorithm_name, False,
                              -1, None, low, high, is_log_distributed, is_integer)


class SearchSpace():
    def __init__(self, json_string):
        self.algorithms = collections.OrderedDict()
        self.pipelines = list()
        for pipeline_json in json_string.values():
            for algo_kv in pipeline_json["_value"]:
                self.pipelines.append(algo_kv["_name"])
                algo = list()
                self.algorithms[algo_kv["_name"]] = algo

                for param_kv in list(algo_kv.items())[1:]:
                    param_name = param_kv[0]
                    param_json = param_kv[1]
                    param_type = param_json["_type"]
                    if param_type == "choice":
                        values = list()
                        for val in param_json["_value"]:
                            values.append(val)
                        algo.append(ParameterRange.categorical(
                            algo_kv["_name"], param_name, values))
                    else:
                        values = param_json["_value"]
                        low = values[0]
                        high = values[1]

                        log = (param_type ==
                               "loguniform" or param_type == "qloguniform")
                        integer = (
                            param_type == "quniform" or param_type == "qloguniform")
                        alog.append(ParameterRange.numerical(
                            algo_kv["_name"], param_name, low, high, log, integer))


class Result():
    def __init__(self, param_id, loss, param):
        self.param_id = param_id
        self.loss = loss
        self.param = param


class TPETuner(Tuner):
    def __init__(self, search_space, minimize_mode=False):
        self.n_startup_jobs = 20
        self.prior_weight = 1.0
        self.oloss_gamma = 0.25
        self.lf = 25
        self.n_ei_candidates = 24
        self.eps = 1e-12
        self.rng = RandomNumberGenerator()

        self.space = SearchSpace(search_space)
        self.minimize = minimize_mode

        self.parameters = dict()  # Dictionary<int, TpeParameters>
        self.running = set()
        self.history = list()
        self.lie = np.inf

    def generate_parameters(self, parameter_id):
        if (len(self.parameters) > self.n_startup_jobs and len(self.running) > 0):
            fake_history = copy.deepcopy(self.history)
            for item in self.running:
                fake_history.append(
                    Result(len(fake_history), self.lie, self.parameters[item]))
            ret, param = self.suggest(self.space, fake_history)
        else:
            ret, param = self.suggest(self.space, self.history)

        self.parameters[parameter_id] = param
        self.running.add(parameter_id)

        return ret

    def update_search_space(self, search_space):
        """
        Update search space definition in tuner by search_space in parameters.

        Will called when first setup experiemnt or update search space in WebUI.
        """
        return SearchSpace(search_space)

    def receive_trial_result(self, parameter_id, loss, **kwargs):
        if not self.minimize:
            loss = - loss
        self.running.remove(parameter_id)
        self.lie = min(self.lie, loss)
        self.history.append(
            Result(parameter_id, loss, self.parameters[parameter_id]))

    def suggest(self, space, history):
        formatted_param = dict()
        param = dict()  # Dictionary<string, double>

        pipeline_index = self.suggest_categorical(
            history, "__pipeline__", len(self.space.pipelines))
        chosen_pipeline = self.space.pipelines[pipeline_index]
        param["__pipeline__"] = pipeline_index

        for algo in self.space.algorithms.items():
            if algo[0] in chosen_pipeline:
                formatted_algo = dict()
                formatted_param[algo[0]] = formatted_algo
                for param_range in algo[1]:
                    if param_range.is_categorical:
                        index = self.suggest_categorical(
                            history, param_range.tag, param_range.size)
                        param[param_range.tag] = index
                        formatted_algo[param_range.name] = param_range.categorical_values[index]
                    else:
                        x = self.suggest_numerical(history, param_range.tag, param_range.low,
                                                   param_range.high, param_range.is_log_distributed,
                                                   param_range.is_integer)
                        param[param_range.tag] = x
                        formatted_algo[param_range.name] = param_range.is_integer if str(
                            int(x)) else str(x)
        # print('formatted_param:{}, param:{}'.format(formatted_param, param))
        return formatted_param, param

    def suggest_categorical(self, history, tag, size):

        if len(history) < self.n_startup_jobs:
            return self.rng.integer(size)

        obs_below, obs_above = self.ap_split_trials(history, tag)

        weights = linear_forgetting_weights(len(obs_below), self.lf)
        counts = bin_count(obs_below, weights, size)
        p = (counts+self.prior_weight)/np.sum(counts+self.prior_weight)
        sample = self.rng.categorical(p, self.n_ei_candidates)
        below_llik = np.log([p[i] for i in sample])

        weights = linear_forgetting_weights(len(obs_above), self.lf)
        counts = bin_count(obs_above, weights, size)
        p = (counts+self.prior_weight)/np.sum(counts+self.prior_weight)
        above_llik = np.log([p[i] for i in sample])

        return self.find_best(sample, below_llik, above_llik)

    def suggest_numerical(self, history, tag, low, high, log, integer):
        if len(history) < self.n_startup_jobs:
            x = self.rng.uniform(low, high)
            if log:
                x = np.exp(x)
            elif integer:
                x = round(x)
            return x

        obs_below, obs_above = self.ap_split_trials(history, tag)

        if log:
            obs_below, obs_above = np.log(obs_below), np.log(obs_above)

        prior_mu = 0.5 * (high + low)
        prior_sigma = high - low

        weights, mus, sigmas = adaptive_parzen_normal(
            obs_below, self.prior_weight, prior_mu, prior_sigma)
        samples = Gmm1(weights, mus, sigmas, low, high, log,
                       integer, self.n_ei_candidates, self.rng)
        below_llik = Gmm1_lpdf(samples, weights, mus,
                               sigmas, low, high, log, integer)

        weights, mus, sigmas = adaptive_parzen_normal(
            obs_below, self.prior_weight, prior_mu, prior_sigma)
        above_llik = Gmm1_lpdf(samples, weights, mus,
                               sigmas, low, high, log, integer)

        return find_best(samples, below_llik, above_llik)

    def ap_split_trials(self, history, tag):
        n_below = min(
            int(np.ceil(self.oloss_gamma * np.sqrt(len(history)))), self.lf)
        history_sorted = sorted(history, key=lambda x: x.loss)
        below = [item for item in history_sorted if tag in item.param.keys()][:n_below]
        above = [item for item in history_sorted if tag in item.param.keys()][n_below:]
        below_value = [item.param[tag] for item in sorted(below, key=lambda x: x.param_id)]
        above_value = [item.param[tag] for item in sorted(above, key=lambda x: x.param_id)]
        return np.asarray(below_value), np.asarray(above_value)

    def find_best(self, samples, below_llik, above_llik):
        best = np.argmax(below_llik - above_llik)
        return samples[best]
