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
        CNode() except +
        CNode(float prior, int action_num, vector[CNode]* ptr_node_pool) except +
        int visit_count, action_num, best_action
        float prior, value_sum
        vector[int] children_index;
        vector[CNode]* ptr_node_pool;

        void expand(vector[float] policy_logits)
        void add_exploration_noise(float exploration_fraction, vector[float] noises)
        float get_mean_q(int isRoot, float parent_q, float discount)

        int expanded()
        float value()
        vector[int] get_trajectory()
        vector[int] get_children_distribution()
        CNode* get_child(int action)

    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int root_num, int action_num, int pool_size) except +
        int root_num, action_num, pool_size
        vector[CNode] roots
        vector[vector[CNode]] node_pools

        void prepare(float root_exploration_fraction, const vector[vector[float]] &noises, const vector[vector[float]] &policies)
        void prepare_no_noise(const vector[vector[float]] &policies)
        void clear()
        vector[vector[int]] get_trajectories()
        vector[vector[int]] get_distributions()
        vector[float] get_values()

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] search_lens
        vector[CNode*] nodes

    cdef void cback_propagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, float value, float discount)
    void cbatch_back_propagate(float discount, vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)
