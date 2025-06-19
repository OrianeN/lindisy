"""
Script to convert a corpus in a standard language (German) into a set of synthetic dialects by applying hierarchical rules.

Requires installation of german_compound_splitter --> https://github.com/repodiac/german_compound_splitter
"""

import argparse
import json
import os
import unicodedata

import epitran
import regex as re
from german_compound_splitter import comp_split
from gliner import GLiNER

from rule_change_applier import RuleChangeApplier

diacritics = {  # Found at https://www.lfsag.unito.it/ipa/index_fr.html
    " ̥", " ̊", "̬", "ʰ", "̹", "̜", "̟", "̠", " ̈", "̽",
    "̩", "̯", "˞", "̤", "̰", "̼", "ʷ", "ʲ", "ˠ", "ˤ", "̴",
    "̝", "̞", "̘", "̙", "̪", "̺", "̻", " ̃", "ⁿ", "ˡ", "̚",
    "͡"  # affricate symbol
}

PH_TEXT = "PH"


class Graphemizer:
    def __init__(self, rules_path):
        self.rules = self.load_rules(rules_path)
        self.phones_regex = re.compile(
            r"\L<phones>",
            phones=list(self.rules)
        )

    def load_rules(self, path):
        rules = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:  # Ignore comments and empty lines
                    continue
                # Clean line from potential inline comments
                line = re.sub(r"\s*#.+$", "", line)
                # Split phoneme and grapheme
                phoneme, grapheme = line.split("/")
                rules[phoneme] = grapheme
        return rules

    def apply_rules(self, word_phones):
        # TODO Apply capitalization (as found in the input (1st/all/none of the letters)
        word_orth = ""
        remaining_phones = word_phones
        while remaining_phones:
            # Look for the next phoneme to replace
            phone_match = self.phones_regex.match(remaining_phones)
            assert phone_match, \
                f"No phone from the rules matched for '{remaining_phones}' (full transcribed word: {word_phones})"
            # Add the corresponding grapheme to the orthographic word
            phone = phone_match[0]
            word_orth += self.rules[phone]
            # Remove graphemized phones from the transcribed word
            remaining_phones = remaining_phones[phone_match.end():]
        return word_orth


class NER:
    def __init__(self, model_name="urchade/gliner_multi-v2.1"):
        self.model = GLiNER.from_pretrained(model_name)
        self.labels = ["named entity", "name", "organization", "work title", "event"]  # TODO Add "web address" (url, email...) ?

    def detect_entities(self, text):
        entities = self.model.predict_entities(text, self.labels)
        # TODO Detect also code-switching ? (or maybe filter out sentences where the LID score is <0.98 ?)
        return entities

    def mask_entities(self, text, ph_text=PH_TEXT):
        # Extract entities
        entities = self.detect_entities(text)
        # Replace spans with ph_text
        text_masked = text
        shift = 0
        for ent in entities:
            ent_start = shift + ent["start"]
            ent_end = shift + ent["end"]
            text_masked = text_masked[:ent_start] + ph_text + text_masked[ent_end:]
            # Compute shift since the start/end positions of entities should be updated as the text is modified
            shift += len(ph_text) - (ent_end - ent_start)
        return text_masked


def load_dialect_rules(dialects_tree, parent_rules=[]):
    dialect2rules = {}
    for node in dialects_tree:
        # Inherit parent rules
        node_rules = parent_rules.copy()
        node_rules.extend(node["rules"])

        if "children" not in node:  # Dialect leaf = end of branch
            dialect2rules[node["id"]] = node_rules
        else:  # (sub)family of dialects
            # Recurse for children nodes/dialects
            dialect2rules.update(load_dialect_rules(
                node["children"],
                parent_rules=node_rules
            ))

    return dialect2rules


def tokenize(line):
    return re.split(r"([[:^alpha:]]+)", line)  # TODO Add '+' to get groups of non-alpha characters ?


def word_needs_transformation(word):
    if word == PH_TEXT:
        return False
    else:
        return bool(re.match(r"[[:alpha:]]", word))


def split_compound(word, ahocs):
    """
    Split and mark compound word.
    :param word: str. word to attempt to split
    :param ahocs: loaded lexicon (ahocs stands for Aho-Corasick search method)
    :return: list of word parts
    """
    try:
        # TODO Silence prints (temporary mockup function ?)
        dissection = comp_split.dissect(word, ahocs, only_nouns=False)  # TODO Decide only_nouns value (True tends to oversplit)
    except IndexError:  # There seems to be a bug with the implementation which raises an error when a word is not found
        return [word]

    if len(dissection) <= 1:
        return [word]
    else:
        return dissection


def phonetize_word(word, phonetizer):
    # Phonetize
    transcription = phonetizer.transliterate(word)
    # Normalize/Simplify diacritics (in phonetic transcription)
    transcription = unicodedata.normalize("NFKD", transcription)
    found_diacritics = diacritics.intersection(set(transcription))
    if found_diacritics:
        transcription = re.sub(rf"[{''.join(found_diacritics)}]", "", transcription)
    return transcription


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("corpus", help="Path to a TXT corpus (one example per line)")
    argparser.add_argument("rules_phonetic",
                           help="Path to the JSON file with phonetic rules for each dialect in a hierarchical tree")
    argparser.add_argument("rules_p2g",
                           help="Path to the TXT file with phoneme-to-grapheme (P2G) rules")
    argparser.add_argument("output", help="Path to the output folder")
    argparser.add_argument("--lang", default="deu-Latn", help="Language code supported by Epitran")
    argparser.add_argument("--lexicon", default="../data/lexicon/german_utf8_linux.dic",
                           help="Path to a German lexicon with one word per line (used for compound words splitting")

    args = argparser.parse_args()

    # Load phonetic transformation rules
    with open(args.rules_phonetic) as f:
        dialects_dict = json.load(f)
        rules_dict = load_dialect_rules(dialects_dict)
    for dialect_id, dialect_rules in rules_dict.items():
        rca = RuleChangeApplier.from_list(dialect_rules)
        rules_dict[dialect_id] = rca

    # Setup file/folder names
    output_folder = args.output
    os.makedirs(output_folder)
    corpus_name = os.path.splitext(os.path.basename(args.corpus))[0]

    # Load external tools: compound splitter, NER model, phonetizer, graphemizer
    ner = NER()
    phonetizer = epitran.Epitran(args.lang)
    graphemizer = Graphemizer(args.rules_p2g)
    ahocs = comp_split.read_dictionary_from_file(args.lexicon)  # ahocs stands for Aho-Corasick search method - used by the compound splitter

    # Phonetize input corpus  # TODO Refactor to go through input corpus only once
    print(f"Generate phonetic transcription of the input corpus (standard)")
    out_path = os.path.join(output_folder, f"{corpus_name}-standard.txt")
    with open(args.corpus) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            # Detect and mask named entities
            line = ner.mask_entities(line)
            # Tokenize
            units = tokenize(line)
            transcribed_line = ""
            for unit in units:
                # Phonetize+graphemize words (but keep punctuation and whitespaces as is)
                if word_needs_transformation(unit):
                    # TODO Count number of words phonetized + filter if <3
                    # Split if compound word
                    unit_splits = split_compound(unit, ahocs)
                    mod_unit = ""
                    for split in unit_splits:
                        split = phonetize_word(split, phonetizer)  # TODO Phonetize only if word is in lexicon ?
                        split = graphemizer.apply_rules(split)
                        mod_unit += split
                    unit = mod_unit
                transcribed_line += unit
            # Write transcribed line to output file
            f_out.write(transcribed_line)

    # Dialectify corpus line by line for each dialect
    for dialect_id in rules_dict.keys():
        print(f"Generate corpus for dialect {dialect_id} (to {out_path})")
        out_path = os.path.join(output_folder, f"{corpus_name}-dialect-{dialect_id}.txt")
        with open(args.corpus) as f_in, open(out_path, "w") as f_out:
            for line in f_in:
                # Detect and mask named entities
                line = ner.mask_entities(line)
                # Tokenize line on all whitespace and punctuation
                units = tokenize(line)
                # Dialectify each unit (token, whitespace, punctuation...)
                dialectified_str = ""
                for unit in units:
                    if word_needs_transformation(unit):
                        # Split if compound word
                        unit_splits = split_compound(unit, ahocs)
                        dialect_unit = ""
                        for split in unit_splits:
                            # Phonetize
                            split = phonetize_word(split, phonetizer)
                            # Apply dialect sound change rule
                            split = rules_dict[dialect_id].apply_changes(split)
                            # Graphemize
                            split = graphemizer.apply_rules(split)
                            dialect_unit += split
                    else:
                        dialect_unit = unit
                    dialectified_str += dialect_unit
                # Write dialectified line to output file
                f_out.write(dialectified_str)

    print(f"Done !")
