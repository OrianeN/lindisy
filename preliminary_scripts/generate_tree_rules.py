"""Script to generate dialect rules along a hierarchical tree"""

import argparse
import json
from copy import deepcopy

from rule_change_applier import RuleChangeApplier


def generate_rules_recursively(dialects_tree, possible_rules):
    level_rules = deepcopy(possible_rules)
    level_rules_selected = []
    for child in dialects_tree:
        # Get rules for upper-level node, making sure that all nodes of the same level+branch are different
        redo_selection = True
        while redo_selection:
            rca = RuleChangeApplier(level_rules, child["nb_rules"])
            if rca.rules_indexes not in level_rules_selected:
                redo_selection = False
        child["rules"] = rca.export_to_list()
        level_rules_selected.append(rca.rules_indexes)
        # Get rules for lower-level nodes
        if child.get("children"):
            # Remove selected rules from possibility to avoid duplicated rules along the same branch
            possible_rules_lower = [r for i, r in enumerate(possible_rules)
                                    if i not in rca.rules_indexes]
            generate_rules_recursively(child["children"], possible_rules_lower)


argparser = argparse.ArgumentParser()
argparser.add_argument("tree", help="JSON file with dialects tree")
argparser.add_argument("rules", help="TXT file with phone change rules")
argparser.add_argument("out", help="Output JSON path")
args = argparser.parse_args()

# Load tree of dialects
with open(args.tree) as f:
    dialects_tree = json.load(f)["hierarchy"]

# Load possible sound change rules
possible_rules = RuleChangeApplier.load_possible_changes(args.rules)

# Generate rules
generate_rules_recursively(dialects_tree, possible_rules)

# Export generated rules
with open(args.out, "w") as f:
    json.dump(dialects_tree, f, indent=4)
