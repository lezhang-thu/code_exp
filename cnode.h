#ifndef CNODE_H
#define CNODE_H

#include "cminimax.h"
#include <cmath>
#include <math.h>
#include <stack>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <time.h>
#include <vector>

namespace tree {

class CNode {
  public:
    int visit_count, action_num, best_action;
    float prior, value_sum;
    std::vector<CNode *> children;

    CNode(float prior, int action_num);
    ~CNode();

    void expand(const std::vector<float> &policy_logits);
    void add_exploration_noise(float exploration_fraction,
                               const std::vector<float> &noises);
    float get_mean_q(int isRoot, float parent_q, float discount);
    void print_out();

    int expanded();

    float value();

    std::vector<int> get_trajectory();
    std::vector<int> get_children_distribution();
    CNode *get_child(int action);
    void release_tree();
};

class CRoots {
  public:
    int root_num, action_num;
    std::vector<CNode *> roots;

    CRoots(int root_num, int action_num);
    ~CRoots();

    void prepare(float root_exploration_fraction,
                 const std::vector<std::vector<float>> &noises,
                 const std::vector<std::vector<float>> &policies);
    void prepare_no_noise(const std::vector<std::vector<float>> &policies);
    std::vector<std::vector<int>> get_trajectories();
    std::vector<std::vector<int>> get_distributions();
    std::vector<float> get_values();
    void update_with_move(int root_idx, int act_idx);
    void release_forest();
};

class CSearchResults {
  public:
    int num;
    std::vector<int> search_lens;
    std::vector<std::vector<CNode *>> search_paths;

    CSearchResults();
    CSearchResults(int num);
    ~CSearchResults();
};

//*********************************************************
void update_tree_q(CNode *root, tools::CMinMaxStats &min_max_stats,
                   float discount);
void cback_propagate(std::vector<CNode *> &search_path,
                     tools::CMinMaxStats &min_max_stats, float value,
                     float discount);
void cbatch_back_propagate(float discount, const std::vector<float> &values,
                           const std::vector<std::vector<float>> &policies,
                           tools::CMinMaxStatsList *min_max_stats_lst,
                           CSearchResults &results);
int cselect_child(CNode *root, tools::CMinMaxStats &min_max_stats,
                  int pb_c_base, float pb_c_init, float discount, float mean_q);
float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats,
                 float parent_mean_q, float total_children_visit_counts,
                 float pb_c_base, float pb_c_init, float discount);
void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init,
                     float discount, tools::CMinMaxStatsList *min_max_stats_lst,
                     CSearchResults &results);
} // namespace tree

#endif
