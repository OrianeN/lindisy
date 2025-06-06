"""Script that selects rules from a set of given rules"""

import argparse
import json

from rule_change_applier import RuleChangeApplier

# Argparse options
argparser = argparse.ArgumentParser()
argparser.add_argument("rules", help="TXT file with phone change rules")
argparser.add_argument("-nr", "--nb_rules", type=int,
                       help="Number of rules to select randomly from all available (default=all)")
argparser.add_argument("-nd", "--nb_dialects", type=int, default=1,
                       help="Number of dialects to generate (1 set of rules per dialect)")
argparser.add_argument("-o", "--output", help="Path to the output JSON file")

args = argparser.parse_args()

# Load possible sound change rules
possible_rules = RuleChangeApplier.load_possible_changes(args.rules)

rules_per_dialect = {}
for i in range(args.nb_dialects):
    # Setup RuleChangeApplier class - this will load the rules and select a subset for this dialect
    rca = RuleChangeApplier(possible_rules, args.nb_rules)
    rules_per_dialect[i] = rca.export_to_list()

# Export selected rules per dialect
with open(args.output, "w") as f:
    json.dump(rules_per_dialect, f, indent=4)
