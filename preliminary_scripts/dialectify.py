
import argparse
import json
import os
import unicodedata

import epitran
from gliner import GLiNER
import regex as re

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
    return re.split(r"([[:^alpha:]])", line)  # TODO Add '+' to get groups of non-alpha characters ?


def word_needs_transformation(word):
    if word == PH_TEXT:
        return False
    else:
        return bool(re.match(r"[[:alpha:]]", word))


# TODO Split and mark compound words (xml tags ?) --> https://github.com/repodiac/german_compound_splitter

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

    # Load NER model, phonetizer, graphemizer
    ner = NER()
    phonetizer = epitran.Epitran(args.lang)
    graphemizer = Graphemizer(args.rules_p2g)

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
                    unit = phonetize_word(unit, phonetizer)
                    unit = graphemizer.apply_rules(unit)
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
                        # Phonetize
                        dialect_unit = phonetize_word(unit, phonetizer)
                        # Apply dialect sound change rule
                        dialect_unit = rules_dict[dialect_id].apply_changes(dialect_unit)
                        # Graphemize
                        dialect_unit = graphemizer.apply_rules(dialect_unit)
                    else:
                        dialect_unit = unit
                    dialectified_str += dialect_unit
                # Write dialectified line to output file
                f_out.write(dialectified_str)

    print(f"Done !")
