"""This module originates from ConlangPLG by Clémentine Fourier."""

import itertools
import json
import random
import re

import sys
sys.path.append("../ConlangPLG/")
from language_definition.phones.vowels import Vowel
from language_definition.phones.pulmonic_consonants import PulmonicConsonant


class RuleChange:

    def __init__(self, orig=None, target=None, ctxt=None):
        if orig is None and None not in (target, ctxt):
            self.target = target
            assert isinstance(ctxt, str), f"ctxt must be of type str, not {type(ctxt)}"
            self.context = re.compile(ctxt)

        else:
            self.target = self.sub_target(
                ctxt, target if isinstance(target, str) else random.choice(target)
            )
            context = self.sub_context(ctxt, orig)

            ctr_v = 0
            while "V" in context:
                context = re.sub(
                    "V",
                    f"(?P<v{str(ctr_v)}>[{''.join([v.ipa for v in Vowel if not v.is_long])}:])",
                    context,
                    count=1,
                )
                self.target = self.target.replace("V", f"\g<v{str(ctr_v)}>", 1)
                ctr_v += 1

            ctr_c = 0
            while "C" in context:
                context = re.sub(
                    "C",
                    f"(?P<c{str(ctr_c)}>[{''.join([c.ipa for c in PulmonicConsonant])}:])",
                    context,
                    count=1,
                )
                self.target = self.target.replace("C", f"\g<c{str(ctr_c)}>", 1)
                ctr_c += 1

            self.context = re.compile(context)

    def sub_context(self, ctxt, to_sub):
        if ctxt == "_":  # replace all
            context = to_sub
        else:
            if ctxt[0] == "#":
                context = "^" + re.sub("_", to_sub, ctxt[1:])
            elif ctxt[-1] == "#":
                context = re.sub("_", to_sub, ctxt[:-1]) + "$"
            else:
                context = re.sub("_", to_sub, ctxt)
        return context

    def sub_target(self, ctxt, to_sub):
        if ctxt == "_":  # replace all
            context = to_sub
        else:
            if ctxt[0] == "#":
                context = re.sub("_", to_sub, ctxt[1:])
            elif ctxt[-1] == "#":
                context = re.sub("_", to_sub, ctxt[:-1])
            else:
                context = re.sub("_", to_sub, ctxt)
        return context

    def __str__(self):
        return str(self.context.pattern) + " => " + self.target

    def __repr__(self):
        return "RuleChange: " + self.__str__()


class RuleChangeApplier:
    def __init__(self, possible_changes: [str,list], num_rules: int = None):
        if isinstance(possible_changes, str):
            self.changes = self.load_possible_changes(possible_changes)
        elif isinstance(possible_changes, list):
            self.changes = possible_changes
        else:
            raise TypeError("possible_changes must be of type str (file path) or list")

        if num_rules is not None:
            self.rules_indexes = random.sample(range(len(self.changes)), num_rules)
            self.changes = list(
                itertools.chain(*[self.changes[i] for i in self.rules_indexes])
            )

    def apply_changes(self, word):
        for rule in self.changes:
            try:
                word = re.sub(rule.context, rule.target, word)
            except re.error:
                raise ValueError(f"Regex issue for rule: {rule}")
        return word

    def apply_changes_to_list(self, word_list):
        result = []
        for word in word_list:
            word = self.apply_changes(word)
            result.append(word)
        return result

    @classmethod
    def load_possible_changes(cls, path):
        possible_changes = []
        with open(path) as f:
            for line_i, line in enumerate(f, start=1):
                cur_change = []
                if line[0] == "!":
                    continue
                try:
                    line, comment = line.split("!!")
                except ValueError:
                    line = line.strip("\n")
                    comment = ""
                rules = line.split("&&")
                for rule in rules:
                    try:
                        src, trg, ctx = rule.split("/")
                    except ValueError as e:
                        raise ValueError(f"Error on rule in line {line_i}")
                    # Handle multiple contexts, src, trg
                    ctx = ctx.split(",")
                    src = src.split(",")
                    trg = trg.split(",")
                    for c in ctx:
                        for s in src:
                            cur_change.append(RuleChange(s, trg, c))
                possible_changes.append(cur_change)
        return possible_changes

    @classmethod
    def from_list(cls, rules):
        changes = []
        for rule in rules:
            context, target = rule.split(";")
            changes.append(RuleChange(ctxt=context, target=target))
        return cls(changes, num_rules=None)

    @classmethod
    def from_file(cls, path, dialect_id=0):
        with open(path) as f:
            rules = json.load(f)[dialect_id]
        return cls.from_list(rules)

    def __repr__(self):
        repr = "\n".join([f" - {rule}" for rule in self.changes])
        return repr

    def export_to_list(self):
        return [f"{rule.context.pattern};{rule.target}" for rule in self.changes]

    def export_to_txt(self, path):
        with open(path, "w") as f:
            for rule_str in self.export_to_list():
                f.write(rule_str + "\n")
