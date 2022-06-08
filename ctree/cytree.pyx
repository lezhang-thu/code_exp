# distutils: language=c++
import ctypes
cimport cython
from ctree cimport CMinMaxStatsList, CRoots, CSearchResults, cbatch_back_propagate, cbatch_traverse
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp.list cimport list as cpplist


cdef class MinMaxStatsList:
    cdef CMinMaxStatsList *cmin_max_stats_lst

    def __cinit__(self, int num):
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    def set_delta(self, float value_delta_max):
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)

    def __dealloc__(self):
        del self.cmin_max_stats_lst


cdef class ResultsWrapper:
    cdef CSearchResults cresults

    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)


cdef class Roots:
    cdef int root_num
    cdef int pool_size
    cdef CRoots *roots

    def __cinit__(self, int root_num, int action_num):
        self.root_num = root_num
        self.roots = new CRoots(root_num, action_num)

    def prepare(self, float root_exploration_fraction, list noises, list policy_logits_pool):
        self.roots[0].prepare(root_exploration_fraction, noises, policy_logits_pool)

    def prepare_no_noise(self, list policy_logits_pool):
        self.roots[0].prepare_no_noise(policy_logits_pool)

    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_values(self):
        return self.roots[0].get_values()

    def update_with_move(self, int root_idx, int act_idx):
        self.roots[0].update_with_move(root_idx, act_idx)

    def release_forest(self):
        self.roots[0].release_forest()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num


def batch_back_propagate(float discount, list values, list policies, MinMaxStatsList min_max_stats_lst, ResultsWrapper results):
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_back_propagate(discount, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults)


def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount, MinMaxStatsList min_max_stats_lst, ResultsWrapper results):
    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults)

