"""Script to generate a tree of dialects"""

import argparse
import json
import random
import uuid


def expand_level_values(value: [list, int], depth: int):
    if isinstance(value, int):
        value = [value]

    if len(value) == depth:
        return value
    elif depth > 1 and len(value) == 1:
        return value * depth  # [1] * 3 --> [1, 1, 1]
    else:
        raise ValueError(f"Values per level incompatible with depth: {value=} vs. {depth=}")


def generate_tree(depth, min_width_per_level, max_width_per_level, min_rules_per_level, max_rules_per_level):
    tree = []
    stats = {
        "nb_dialects": 0,
        "nb_nodes_per_level": [0] * (depth-1)
    }

    # Select number of child nodes to generate
    min_width = min_width_per_level[0]
    max_width = max_width_per_level[0]
    nb_children = random.choice(range(min_width, max_width + 1))
    if depth > 1:
        stats["nb_nodes_per_level"][0] = nb_children

    # Generate child dialects
    for child_i in range(nb_children):
        # Create node
        node = {"id": str(uuid.uuid4()).split("-")[0]}
        # Define number of rules that should be assigned to this dialect
        min_rules = min_rules_per_level[0]
        max_rules = max_rules_per_level[0]
        nb_rules = random.choice(range(min_rules, max_rules + 1))
        node["nb_rules"] = nb_rules

        # Generate grand-children dialects
        if depth > 1:
            node["children"], child_stats = generate_tree(
                depth=depth-1,
                min_width_per_level=min_width_per_level[1:],
                max_width_per_level=max_width_per_level[1:],
                min_rules_per_level=min_rules_per_level[1:],
                max_rules_per_level=max_rules_per_level[1:]
            )
            stats["nb_dialects"] += child_stats["nb_dialects"]
            for depth_i in range(1, depth-1):
                stats["nb_nodes_per_level"][depth_i] += child_stats["nb_nodes_per_level"][depth_i-1]
        else:
            stats["nb_dialects"] += 1
        # Append node to the tree
        tree.append(node)
    return tree, stats


argparser = argparse.ArgumentParser()

argparser.add_argument("out", help="Path to the output JSON file with the generated tree")
argparser.add_argument("-d", "--depth", type=int, default=3,
                       help="Depth of the tree")
argparser.add_argument("-wb", "--min_width", type=int, default=[2, 1, 1], nargs="+",
                       help="Minimum number of nodes per level")
argparser.add_argument("-wu", "--max_width", type=int, default=[4, 4, 4], nargs="+",
                       help="Maximum number of nodes per level (included)")
argparser.add_argument("-rb", "--min_rules", type=int, default=1, nargs="+",
                       help="Minimum number of rules per level")
argparser.add_argument("-ru", "--max_rules", type=int, default=2, nargs="+",
                       help="Maximum number of rules per level (included)")

args = argparser.parse_args()

# Expand level values (e.g. if depth 3 and min_width [2], make min_width [2, 2, 2])
args.min_width = expand_level_values(args.min_width, args.depth)
args.max_width = expand_level_values(args.max_width, args.depth)
args.min_rules = expand_level_values(args.min_rules, args.depth)
args.max_rules = expand_level_values(args.max_rules, args.depth)

# Generate the tree of dialects
tree, stats = generate_tree(args.depth, args.min_width, args.max_width, args.min_rules, args.max_rules)
print(stats)
tree = {
    **stats,
    "hierarchy": tree
}

# Output to JSON
with open(args.out, "w") as f:
    json.dump(tree, f, indent=4)
