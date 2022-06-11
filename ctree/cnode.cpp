#include "cnode.h"
#include <iostream>

namespace tree {

CSearchResults::CSearchResults() { this->num = 0; }

CSearchResults::CSearchResults(int num) {
    this->num = num;
    for (int i = 0; i < num; ++i) {
        this->search_paths.push_back(std::vector<CNode *>());
    }
}

CSearchResults::~CSearchResults() {}

//*********************************************************

CNode::CNode(float prior, int action_num) {
    this->visit_count = 0;
    this->action_num = action_num;
    this->best_action = -1;

    this->prior = prior;
    this->value_sum = 0;
}

CNode::~CNode() {}

void CNode::expand(const std::vector<float> &policy_priors) {
    int action_num = this->action_num;
    this->children.reserve(action_num);
    for (int a = 0; a < action_num; ++a) {
        this->children.push_back(new CNode(policy_priors[a], action_num));
    }
}

void CNode::add_exploration_noise(float exploration_fraction,
                                  const std::vector<float> &noises) {
    float noise, prior;
    for (int a = 0; a < this->action_num; ++a) {
        noise = noises[a];
        CNode *child = this->get_child(a);

        prior = child->prior;
        child->prior =
            prior * (1 - exploration_fraction) + noise * exploration_fraction;
    }
}

float CNode::get_mean_q(float parent_q, float discount) {
    float total_unsigned_q = 0.0;
    int total_visits = 0;
    for (int a = 0; a < this->action_num; ++a) {
        CNode *child = this->get_child(a);
        if (child->visit_count > 0) {
            float qsa = discount * child->value();
            total_unsigned_q += qsa;
            total_visits += 1;
        }
    }
    return (parent_q + total_unsigned_q) / (1 + total_visits);
}

void CNode::print_out() { return; }

int CNode::expanded() { return this->children.size() > 0; }

float CNode::value() {
    return this->visit_count == 0 ? 0.0 : this->value_sum / this->visit_count;
}

std::vector<int> CNode::get_trajectory() {
    std::vector<int> traj;

    CNode *node = this;
    int best_action = node->best_action;
    while (best_action >= 0) {
        traj.push_back(best_action);

        node = node->get_child(best_action);
        best_action = node->best_action;
    }
    return traj;
}

std::vector<int> CNode::get_children_distribution() {
    std::vector<int> distribution;
    if (this->expanded()) {
        for (int a = 0; a < this->action_num; ++a) {
            CNode *child = this->get_child(a);
            distribution.push_back(child->visit_count);
        }
    }
    return distribution;
}

CNode *CNode::get_child(int action) { return this->children[action]; }

void CNode::release_tree() {
    std::stack<CNode *> node_stack;
    node_stack.push(this);
    while (node_stack.size() > 0) {
        CNode *node = node_stack.top();
        node_stack.pop();

        if (node->expanded()) {
            for (int a = 0; a < node->action_num; ++a) {
                CNode *child = node->get_child(a);
                if (child != nullptr)
                    node_stack.push(child);
            }
        }
        delete node;
    }
}

//*********************************************************

CRoots::CRoots(int root_num, int action_num) {
    this->root_num = root_num;
    this->action_num = action_num;
    this->roots.reserve(root_num);

    for (int i = 0; i < root_num; ++i) {
        // REFER:
        // https://github.com/lezhang-thu/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py#L101
        this->roots.push_back(new CNode(1.0, action_num));
    }
}

CRoots::~CRoots() {}

void CRoots::prepare(float root_exploration_fraction,
                     const std::vector<std::vector<float>> &noises,
                     const std::vector<std::vector<float>> &policies) {
    for (int i = 0; i < this->root_num; ++i) {
        this->roots[i]->expand(policies[i]);
        this->roots[i]->add_exploration_noise(root_exploration_fraction,
                                              noises[i]);
        this->roots[i]->visit_count += 1;
    }
}

void CRoots::prepare_no_noise(const std::vector<std::vector<float>> &policies) {
    for (int i = 0; i < this->root_num; ++i) {
        this->roots[i]->expand(policies[i]);
        this->roots[i]->visit_count += 1;
    }
}

std::vector<std::vector<int>> CRoots::get_trajectories() {
    std::vector<std::vector<int>> trajs;
    trajs.reserve(this->root_num);

    for (int i = 0; i < this->root_num; ++i) {
        trajs.push_back(this->roots[i]->get_trajectory());
    }
    return trajs;
}

std::vector<std::vector<int>> CRoots::get_distributions() {
    std::vector<std::vector<int>> distributions;
    distributions.reserve(this->root_num);

    for (int i = 0; i < this->root_num; ++i) {
        distributions.push_back(this->roots[i]->get_children_distribution());
    }
    return distributions;
}

// MIGHT BE of NO USE
// captioning seq. is of len 20
// MIGHT use the final CIDEr score
std::vector<float> CRoots::get_values() {
    std::vector<float> values;
    for (int i = 0; i < this->root_num; ++i) {
        values.push_back(this->roots[i]->value());
    }
    return values;
}

// REFER:
// https://github.com/lezhang-thu/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py#L157
void CRoots::update_with_move(int root_idx, int act_idx) {
    CNode *root = this->roots[root_idx];
    this->roots[root_idx] = root->get_child(act_idx);
    root->children[act_idx] = nullptr;
    root->release_tree();
}

void CRoots::release_forest() {
    for (int i = 0; i < this->root_num; ++i)
        this->roots[i]->release_tree();
}

//*********************************************************

void update_tree_q(CNode *root, tools::CMinMaxStats &min_max_stats,
                   float discount) {
    // only consider `node->value()` for all internal nodes
    std::stack<CNode *> node_stack;
    node_stack.push(root);
    while (node_stack.size() > 0) {
        CNode *node = node_stack.top();
        node_stack.pop();

        if (node != root) {
            float qsa = discount * node->value();
            min_max_stats.update(qsa);
        }

        for (int a = 0; a < node->action_num; ++a) {
            CNode *child = node->get_child(a);
            if (child->expanded()) {
                node_stack.push(child);
            }
        }
    }
}

void cback_propagate(std::vector<CNode *> &search_path,
                     tools::CMinMaxStats &min_max_stats, float value,
                     float discount) {
    for (int i = search_path.size() - 1; i >= 0; --i) {
        CNode *node = search_path[i];
        node->value_sum += value;
        node->visit_count += 1;

        value *= discount;
    }
    min_max_stats.clear();
    CNode *root = search_path[0];
    update_tree_q(root, min_max_stats, discount);
}

void cbatch_back_propagate(float discount, const std::vector<float> &values,
                           const std::vector<std::vector<float>> &policies,
                           tools::CMinMaxStatsList *min_max_stats_lst,
                           CSearchResults &results) {
    for (int i = 0; i < results.num; ++i) {
        // NOT end of game
        // REFER:
        // https://github.com/lezhang-thu/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py#L119-L137
        if (policies[i][0] != -1.0)
            results.search_paths[i].back()->expand(policies[i]);
        cback_propagate(results.search_paths[i],
                        min_max_stats_lst->stats_lst[i], values[i], discount);
    }
}

int cselect_child(CNode *root, tools::CMinMaxStats &min_max_stats,
                  int pb_c_base, float pb_c_init, float discount,
                  float mean_q) {
    float max_score = FLOAT_MIN;
    const float epsilon = 1e-6;
    std::vector<int> max_index_lst;
    for (int a = 0; a < root->action_num; ++a) {
        CNode *child = root->get_child(a);
        // should be `root->visit_count - 1`
        // in AlphaZero it is `parent.visit_count`
        float temp_score =
            cucb_score(child, min_max_stats, mean_q, root->visit_count - 1,
                       pb_c_base, pb_c_init, discount);

        if (max_score < temp_score) {
            max_score = temp_score;

            max_index_lst.clear();
            max_index_lst.push_back(a);
        } else if (temp_score >= max_score - epsilon) {
            max_index_lst.push_back(a);
        }
    }

    int action = 0;
    if (max_index_lst.size() > 0) {
        action = max_index_lst[rand() % max_index_lst.size()];
    }
    return action;
}

float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats,
                 float parent_mean_q, float total_children_visit_counts,
                 float pb_c_base, float pb_c_init, float discount) {
    float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
    pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) +
           pb_c_init;
    pb_c *= sqrt(total_children_visit_counts) / (child->visit_count + 1);

    prior_score = pb_c * child->prior;
    if (child->visit_count == 0) {
        value_score = parent_mean_q;
    } else {
        value_score = discount * child->value();
    }

    value_score = min_max_stats.normalize(value_score);
    if (value_score < 0)
        value_score = 0;
    if (value_score > 1)
        value_score = 1;

    float ucb_value = prior_score + value_score;
    return ucb_value;
}

void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init,
                     float discount, tools::CMinMaxStatsList *min_max_stats_lst,
                     CSearchResults &results) {
    // set seed
    timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec);

    for (int i = 0; i < results.num; ++i) {
        CNode *node = roots->roots[i];
        results.search_paths[i].push_back(node);

        // invariant: `mean_q` is \hat{Q}(s) for `node`
        float mean_q = 0.0;
        while (node->expanded()) {
            int action = cselect_child(node, min_max_stats_lst->stats_lst[i],
                                       pb_c_base, pb_c_init, discount, mean_q);
            node->best_action = action;
            // next
            node = node->get_child(action);
            // node always non-root
            mean_q = node->get_mean_q(mean_q, discount);
            results.search_paths[i].push_back(node);
        }
        // loop invariant:
        // after `while`, `results.search_paths[i]`: a root-to-leaf path
        // `value_sum / visit_count` is v(s) for the node (i.e. state s)
    }
}

} // namespace tree