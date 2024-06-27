# IMPORTS
import numpy as np
import networkx as nx
from functools import reduce

# function to compute the adjacency matrix
def get_adj(var_rel, rule_rel):
    n = var_rel.shape[1]
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v = var_rel[:, i] * var_rel[:, j] * rule_rel
            adj[i, j] = 1 - reduce(lambda x, y: x * y, 1 - v, 1)
    return adj/adj.sum().sum()*100

# functions to compute the feature relevance matrices and the rule relevance matrices
def get_relevances(rules, train, feat_names):
    rule_rels = {}
    var_rels = {}
    for rule_type in ["support", "confidence", "lift", "relevance", "equal"]:
        rule_rels[rule_type] = np.array([get_rule_rel(train, rule, rule_type) for rule in rules])
    for cond_type in ["impurity", "relevance"]:
        varr = np.zeros((len(rules), len(feat_names)))    
        for i, rule in enumerate(rules):
            for j, var in enumerate(feat_names):
                if var in [cond["variable"] for cond in rule["conditions"]]:
                    varr[i,j] =  get_var_rel(train, var, rule, cond_type)
        var_rels[cond_type] = varr
    return rule_rels, var_rels

# evaluate contribution of the rule to the class as relevance / support / confidence / lift
def get_rule_rel(df_train, rule, type="support"):
    A = df_train.iloc[:, :-1].apply(lambda row: all(apply_condition(cond["operation"], row[cond["variable"]], cond["threshold"])
                                                                                        for cond in rule['conditions']), axis=1)
    B = (df_train["target"] == rule['outcome'])
    #
    support = sum(A & B) / len(df_train)
    confidence = sum(A & B) / sum(A) if sum(A) != 0 else 0
    lift = confidence / (sum(B) / len(df_train)) if sum(B) != 0 else 0

    covering = sum(A & B) / sum(B)
    error = sum(A & ~B) / sum(~B)
    if type == "support":
        return support
    elif type == "confidence":
        return confidence
    elif type == "lift":
        return lift
    elif type == "relevance":
        return covering*(1-error)
    elif type == "equal":
        return 1
    else:
        return np.nan
    
# evaluate contribution of the feature to the rule as relevance / impurity
def get_var_rel(df_train, var, rule, type="impurity"):
    #
    conds = rule['conditions']
    cond_wo = [cond for cond in rule["conditions"] if cond["variable"]!= var]
    #cond_flip = cond_wo + [flip_condition(cond)]    
    #
    if type == "impurity":
        parentlabels = list(df_train[df_train.iloc[:, :-1].apply(lambda row: 
            all(apply_condition(cond["operation"], row[cond["variable"]], cond["threshold"]) for cond in cond_wo), axis=1)]["target"])
        rulelabels = list(df_train[df_train.iloc[:, :-1].apply(lambda row: 
            all(apply_condition(cond["operation"], row[cond["variable"]], cond["threshold"]) for cond in conds), axis=1)]["target"])
        otherlabels = subtract_lists(parentlabels, rulelabels)
        #otherlabels = list(df_train[df_train.iloc[:, :-1].apply(lambda row: 
        #    all(apply_condition(cond["operation"], row[cond["variable"]], cond["threshold"]) for cond in cond_flip), axis=1)]["target"])
        return calculate_impurity_gain(parentlabels, rulelabels, otherlabels)
        
    elif type == "relevance":
        A = df_train.iloc[:, :-1].apply(lambda row: all(apply_condition(cond["operation"], row[cond["variable"]], cond["threshold"]) 
                                                                                            for cond in conds), axis=1)
        A1 = df_train.iloc[:, :-1].apply(lambda row: all(apply_condition(cond["operation"], row[cond["variable"]], cond["threshold"]) 
                                                                                            for cond in cond_wo), axis=1)
        B = (df_train["target"] == rule['outcome'])
        nB = ~B
        #
        error_mod = sum(A1 & nB) / sum(nB)
        error = sum(A & nB) / sum(nB)
        covering = sum(A & B) / sum(B)
        return (error_mod - error) * covering
    else:
        return np.nan
    
def apply_condition(operation, value, threshold):
    if operation == "<=":
        return value <= threshold
    if operation == "=<":
        return value <= threshold
    elif operation == "<":
        return value < threshold
    if operation == ">=":
        return value >= threshold
    if operation == "=>":
        return value >= threshold
    elif operation == ">":
        return value > threshold
    
def subtract_lists(list1, list2):
    result = list1.copy() 
    for element in list2:
        if element in result:
            result.remove(element) 
    return result

def calculate_impurity_gain(original_labels, left_labels, right_labels):
    try:
        original_impurity = gini_impurity(original_labels)
        left_impurity = gini_impurity(left_labels)
        right_impurity = gini_impurity(right_labels)
        n_samples = len(original_labels)
        impurity_after = (len(left_labels) / n_samples) * left_impurity + (len(right_labels) / n_samples) * right_impurity
        impurity_gain = original_impurity - impurity_after
        return impurity_gain
    except:
        return 0
    
def gini_impurity(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities ** 2)

def flip_condition(condition):
    opposite_operators = {'<': '>=', '<=': '>', '>': '<=', '>=': '<', '==': '!=', '!=': '=='}
    return {'variable': condition['variable'], 'operation': opposite_operators[condition['operation']], 'threshold': condition['threshold']}


# plot graph
def nudge(pos, x_shift, y_shift):
    return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

def plot_adj_graph(adj, ax, feat_names=None, color="gold", nudgeval=0.1):
    adj1 = adj.copy()
    centrality = adj1.sum(axis=1)
    np.fill_diagonal(adj1, 0)
    
    # Create the graph
    G = nx.from_numpy_array(adj1)
    
    node_sizes = [15 * centrality[node] for node in G.nodes()]
    
    if feat_names is None:
        labels = {i: i for i in G.nodes()}
    else:
        labels = {i: feat_names[i] for i in G.nodes()}
    
    pos = nx.circular_layout(G)  # positions for all nodes
    pos_labels = nudge(pos, 0, nudgeval) 
    
    # Plot nodes
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=np.array(node_sizes), node_color=color)
    nodes.set_edgecolor('dimgray')
    
    # Plot edges if there are any
    if G.number_of_edges() > 0:
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, width=np.array(weights), edge_color='gray', alpha=0.4)
        #nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, width=np.log(np.array(weights)+1)*4, edge_color='gray', alpha=0.4)
    
    # Plot labels
    nx.draw_networkx_labels(G, pos_labels, labels, font_size=10, font_family="sans-serif", ax=ax)
