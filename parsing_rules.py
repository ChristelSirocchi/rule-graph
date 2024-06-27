# Imports
import re
import numpy as np
from psyke import EvaluableModel
from sklearn.tree import _tree

# functions to print scores for the rules extracted using PsyKE
def print_scores(scores):
    print(f'Classification accuracy = {scores[EvaluableModel.ClassificationScore.ACCURACY][0]:.2f} (data), '
          f'{scores[EvaluableModel.ClassificationScore.ACCURACY][1]:.2f} (BB)\n'
          f'F1 = {scores[EvaluableModel.ClassificationScore.F1][0]:.2f} (data), '
          f'{scores[EvaluableModel.ClassificationScore.F1][1]:.2f} (BB)')

def get_scores(extractor, test, predictor):
    return extractor.score(test, predictor, True, True, False, task=EvaluableModel.Task.CLASSIFICATION,
                           scoring_function=[EvaluableModel.ClassificationScore.ACCURACY,
                                             EvaluableModel.ClassificationScore.F1])


# PARSING FUNCTIONS
# function to parse rules from PsyKE extracted theories
def parse_rules(rule_string):
    opposite_signs = {
        '<': '>',
        '>': '<',
        '<=': '>=',
        '>=': '<=',
        '=<': '=>',
        '=>': '=<',
    }
    rules = []
    # predicate with rules
    rule_pattern = re.compile(r"target\((.*?)\)\s:-\s*(.*?)\.($|\s)")
    # final predicate with no rules
    rule_pattern_end = re.compile(r"target\((.*?)\)\.")
    # rule condition
    condition_pattern = re.compile(r'([a-zA-Z]+)\s*([=<>]+)\s*([0-9.-]+)')
    # rule interval
    interval_pattern = re.compile(r"([a-zA-Z]+)\s+in\s+\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]")
    for match in rule_pattern.finditer(rule_string):
        outcome = match.group(1).strip().split()[-1].strip("\"'")
        conditions_str = match.group(2).strip()
        conditions = []
        # condition with less than / greater than
        for condition_match in condition_pattern.finditer(conditions_str):
            variable = condition_match.group(1)
            operation = condition_match.group(2)
            threshold = float(condition_match.group(3))
            conditions.append({
                "variable": variable,
                "operation": operation,
                "threshold": threshold
            })
        # condition with interval
        for condition_match in interval_pattern.finditer(conditions_str):
            variable = condition_match.group(1)
            lower_bound = float(condition_match.group(2))
            upper_bound = float(condition_match.group(3))
            conditions.extend([
                {"variable": variable,
                "operation": ">=",
                "threshold": lower_bound},
                {"variable": variable,
                "operation": "<=",
                "threshold": upper_bound}           
            ]) 
        rules.append({
            "conditions": conditions,
            "outcome": outcome
        })
    # rule with no conditions
    for match in rule_pattern_end.finditer(rule_string):
        outcome = match.group(1).strip().split()[-1]
        rules.append({
            "conditions": [],
            "outcome": outcome
        })
    return rules

# function to parse rules from RIPPER rule output
def parse_ripper_rules(rules_str, feat_names, pos_class):
    rules_list = rules_str.replace('[', '').replace(']', '').split(' V ')
    parsed_rules = []

    for rule_str in rules_list:
        conditions = rule_str.split('^')
        parsed_conditions = []

        for condition in conditions:
            if '=>' in condition:
                # Greater than or equal condition
                feat_num, threshold = condition.split('=>')
                feat_name = feat_names[int(feat_num)]
                threshold = float(threshold)
                parsed_conditions.append({'variable': feat_name, 'operation': '>=', 'threshold': threshold})
            elif '=<' in condition:
                # Less than or equal condition
                feat_num, threshold = condition.split('=<')
                feat_name = feat_names[int(feat_num)]
                threshold = float(threshold)
                parsed_conditions.append({'variable': feat_name, 'operation': '<=', 'threshold': threshold})
            elif '=' in condition:
                if '-' in condition:
                    # Interval condition
                    feat_num, interval = condition.split('=')
                    feat_name = feat_names[int(feat_num)]
                    lower, upper = map(float, interval.split('-'))
                    parsed_conditions.append({'variable': feat_name, 'operation': '>=', 'threshold': lower})
                    parsed_conditions.append({'variable': feat_name, 'operation': '<=', 'threshold': upper})
                else:
                    # Equality condition
                    feat_num, value = condition.split('=')
                    feat_name = feat_names[int(feat_num)]
                    value = float(value)
                    parsed_conditions.append({'variable': feat_name, 'operation': '==', 'threshold': value})
        parsed_rules.append({'conditions': parsed_conditions, 'outcome': pos_class})  # Assuming 'True' for now

    return parsed_rules

# function to parse rules from LLM rule output
def parse_llm_rules(rule):
    # Initialize the output dictionary
    parsed_rule = {'conditions': [], 'outcome': None}
    opposite_operators = {'<': '>=', '<=': '>', '>': '<=', '>=': '<', '==': '!=', '!=': '=='}
    # Regex to match conditions and the outcome
    condition_pattern = re.compile(r'(\w+) (<=|<|>|>=|=) (-?[0-9.]+)')
    interval_pattern = re.compile(r'(-?[0-9.]+) (<|<=) (\w+) (<|<=) (-?[0-9.]+)')
    outcome_pattern = re.compile(r'target in \{(\w+)\}')
    
    # Find all interval conditions
    intervals = interval_pattern.findall(rule)
    for interval in intervals:
        lower_bound, lower_op, variable, upper_op, upper_bound = interval
        parsed_rule['conditions'].append({
            'variable': variable,
            'operation': opposite_operators[lower_op],
            'threshold': float(lower_bound)
        })
        parsed_rule['conditions'].append({
            'variable': variable,
            'operation': upper_op,
            'threshold': float(upper_bound)
        })
    
    # Find all regular conditions
    conditions = condition_pattern.findall(rule)
    for cond in conditions:
        variable, operation, threshold = cond
        parsed_rule['conditions'].append({
            'variable': variable,
            'operation': operation,
            'threshold': float(threshold)
        })
    
    # Remove duplicates (cases where the interval created two similar conditions)
    parsed_rule['conditions'] = [dict(t) for t in {tuple(d.items()) for d in parsed_rule['conditions']}]
    
    # Find the outcome
    outcome_match = outcome_pattern.search(rule)
    if outcome_match:
        parsed_rule['outcome'] = outcome_match.group(1)
    
    return parsed_rule

# Function to parse rules from Decision Tree
def traverse_tree(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # Internal node
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_conditions = conditions + [{
                'variable': name, 
                'operation': '<=', 
                'threshold': threshold
            }]
            right_conditions = conditions + [{
                'variable': name, 
                'operation': '>', 
                'threshold': threshold
            }]
            recurse(tree_.children_left[node], left_conditions)
            recurse(tree_.children_right[node], right_conditions)
        else:  # Leaf node
            value = tree_.value[node]
            outcome_index = np.argmax(value)  # Class with the highest count
            outcome = class_names[outcome_index]  # Map index to class name
            rules.append({
                'conditions': conditions,
                'outcome': outcome
            })

    rules = []
    recurse(0, [])
    return rules