# distutils: language=c++
from libcpp.vector cimport vector


cdef extern from "cminimax.cpp":
    pass


cdef extern from "cminimax.h" namespace "tools":
    cdef cppclass CMinMaxStats:
        CMinMaxStats() except +
        float maximum, minimum, value_delta_max

        void set_delta(float value_delta_max)
        void update(float value)
        void clear()
        float normalize(float value)

    cdef cppclass CMinMaxStatsList:
        CMinMaxStatsList() except +
        CMinMaxStatsList(int num) except +
        int num
        vector[CMinMaxStats] stats_lst

        void set_delta(float value_delta_max)

cdef extern from "cnode.cpp":
    pass


cdef extern from "cnode.h" namespace "tree":
    cdef cppclass CNode:
        pass

    cdef cppclass CRoots:
        CRoots(int root_num, int action_num) except +
        int root_num, action_num
        vector[CNode*] roots

        void prepare(float root_exploration_fraction, const vector[vector[float]] &noises, const vector[vector[float]] &policies)
        void prepare_no_noise(const vector[vector[float]] &policies)
        vector[vector[int]] get_trajectories()
        vector[vector[int]] get_distributions()
        vector[float] get_values()
        void update_with_move(int root_idx, int act_idx)
        void release_forest()

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] search_lens

    cdef void cback_propagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, float value, float discount)
    void cbatch_back_propagate(float discount, vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)
