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

UNIT_TYPE_ALPHA_LOWER = "alpha"
UNIT_TYPE_ALPHA_CAPITALIZED = "Alpha"
UNIT_TYPE_ALPHA_UPPER = "ALPHA"
UNIT_TYPE_ALPHA = UNIT_TYPE_ALPHA_LOWER
UNIT_TYPE_NON_ALPHA = "other"


def load_sound_rules(path_sound_rules: str):
    # Load phonetic transformation rules
    with open(path_sound_rules) as f:
        dialects_dict = json.load(f)
        rules_dict = load_dialect_rules(dialects_dict)
    for dialect_id, dialect_rules in rules_dict.items():
        rca = RuleChangeApplier.from_list(dialect_rules)
        rules_dict[dialect_id] = rca
    return rules_dict


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


class Phonetizer:
    DIACRITICS = {  # Found at https://www.lfsag.unito.it/ipa/index_fr.html
        " ̥", " ̊", "̬", "ʰ", "̹", "̜", "̟", "̠", " ̈", "̽",
        "̩", "̯", "˞", "̤", "̰", "̼", "ʷ", "ʲ", "ˠ", "ˤ", "̴",
        "̝", "̞", "̘", "̙", "̪", "̺", "̻", " ̃", "ⁿ", "ˡ", "̚",
        "͡"  # affricate symbol
    }

    def __init__(self, lang="deu-Latn"):
        self.epi = epitran.Epitran(lang)

    def __call__(self, word):
        transcription = self.epi.transliterate(word)
        transcription = self.normalize_transcription(transcription)
        return transcription

    def normalize_transcription(self, transcription):
        # Normalize encoding
        transcription = unicodedata.normalize("NFKD", transcription)
        # Remove all diacritics
        found_diacritics = self.DIACRITICS.intersection(set(transcription))
        if found_diacritics:
            transcription = re.sub(rf"[{''.join(found_diacritics)}]", "", transcription)
        # Normalize some sounds that look like normal letters
        transcription = re.sub("ɡ", "g", transcription)
        transcription = re.sub("ç", "ç", transcription)
        return transcription


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
    PH_TEXT = "PH"  # TODO Use pseudo-XML instead (<ph>entity</ph>)

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


class Dialectifier:

    def __init__(self, path_sound_rules: str, path_p2g: str, path_lexicon: str,
                 output_folder: str, corpus_name: str, lang="deu-Latn", min_words_per_line=3):
        self.sound_rules = load_sound_rules(path_sound_rules)
        self.min_words_per_line = min_words_per_line

        # Load external tools: compound splitter, NER model, phonetizer, graphemizer
        self.ner = NER()
        self.phonetizer = Phonetizer(lang)
        self.graphemizer = Graphemizer(path_p2g)
        self.ahocs = comp_split.read_dictionary_from_file(path_lexicon)  # ahocs stands for Aho-Corasick search method - used by the compound splitter

        # Setup paths
        self.output_folder = output_folder
        self.corpus_name = corpus_name
        self.corpus_prep_path = None

    def preprocess_corpus(self, corpus_path: str):
        self.corpus_prep_path = self.make_output_path("preprocessed")
        with open(corpus_path) as f_in, open(self.corpus_prep_path, "w") as f_out:
            for line in f_in:
                line_units = []
                # Detect and mask named entities
                line = self.ner.mask_entities(line)
                # Tokenize
                units = self.tokenize(line)
                nb_words = 0
                for unit in units:
                    # Phonetize+graphemize words (but keep punctuation and whitespaces as is)
                    if self.word_needs_transformation(unit):
                        # Detect type of capitalization
                        unit_type = self.get_case_type(unit)
                        # Split if compound word
                        unit_splits = self.split_compound(unit)
                        mod_splits = []
                        for split in unit_splits:
                            split = self.phonetizer(split)
                            mod_splits.append(split)
                        line_units.append((unit_type, mod_splits))
                        nb_words += 1
                    else:
                        line_units.append((UNIT_TYPE_NON_ALPHA, unit))
                # Make line empty if too few phonetized words
                if self.min_words_per_line and nb_words < self.min_words_per_line:
                    line_units = [[UNIT_TYPE_NON_ALPHA, "\n"]]
                # Write transcribed line to output file
                f_out.write(json.dumps(line_units)+"\n")

    def generate_variety_corpus(self, dialect_id: [str, None], out_path: str):
        with open(self.corpus_prep_path) as f_prep, open(out_path, "w") as f_out:
            for line in f_prep:
                transcribed_line = ""
                units = json.loads(line.strip())
                for unit_type, unit_val in units:
                    if unit_type.lower() == UNIT_TYPE_ALPHA:
                        assert isinstance(unit_val, list)
                        merged_splits = ""
                        for split in unit_val:
                            if dialect_id is not None:
                                # Apply dialect sound change rules
                                split = self.sound_rules[dialect_id].apply_changes(split)
                            # Graphemize sounds
                            split = self.graphemizer.apply_rules(split)
                            merged_splits += split
                        unit_val = merged_splits
                        # Reapply capitalization as original
                        unit_val = self.apply_case(unit_val, unit_type)
                    transcribed_line += unit_val
                # Write transcribed line to output file
                f_out.write(transcribed_line)

    def generate_all_varieties(self):
        # Generate corpus for standard variety
        out_path = self.make_output_path("standard")
        print(f"Generate corpus for the standard variety (to {out_path})")
        self.generate_variety_corpus(dialect_id=None, out_path=out_path)

        # Generate all dialects
        for dialect_id in self.sound_rules.keys():
            print(f"Generate corpus for dialect {dialect_id} (to {out_path})")
            out_path = self.make_output_path(f"dialect-{dialect_id}")
            self.generate_variety_corpus(dialect_id, out_path)

    def make_output_path(self, suffix):
        return os.path.join(self.output_folder, f"{self.corpus_name}-{suffix}.txt")

    @staticmethod
    def tokenize(line):
        return re.split(r"([[:^alpha:]]+)", line)

    @staticmethod
    def word_needs_transformation(word):
        if word == NER.PH_TEXT:
            return False
        else:
            return bool(re.match(r"[[:alpha:]]", word))

    def split_compound(self, word):
        """
        Split and mark compound word.
        :param word: str. word to attempt to split
        :param ahocs: loaded lexicon (ahocs stands for Aho-Corasick search method)
        :return: list of word parts
        """
        try:
            # TODO Silence prints (temporary mockup function ?)
            dissection = comp_split.dissect(word, self.ahocs, only_nouns=False)  # TODO Decide only_nouns value (True tends to oversplit)
        except IndexError:  # An error is raised when a word is not found in the lexicon
            return [word]

        if len(dissection) <= 1:
            return [word]
        else:
            return dissection

    @staticmethod
    def get_case_type(word: str):
        if word == word.upper():
            return UNIT_TYPE_ALPHA_UPPER
        elif word == word.capitalize():
            return UNIT_TYPE_ALPHA_CAPITALIZED
        else:
            return UNIT_TYPE_ALPHA_LOWER

    @staticmethod
    def apply_case(word, case_type):
        if case_type == UNIT_TYPE_ALPHA_LOWER:
            return word
        elif case_type == UNIT_TYPE_ALPHA_CAPITALIZED:
            return word.capitalize()
        elif case_type == UNIT_TYPE_ALPHA_UPPER:
            return word.upper()
        else:
            raise ValueError(f"Case type not recognized: {case_type}")


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

    # Setup file/folder names
    output_folder = args.output
    os.makedirs(output_folder)  # TODO Auto-increment ?
    corpus_name = os.path.splitext(os.path.basename(args.corpus))[0]

    # Initialize Dialectifier
    dialectifier = Dialectifier(
        path_sound_rules=args.rules_phonetic,
        path_p2g=args.rules_p2g,
        path_lexicon=args.lexicon,
        output_folder=output_folder,
        corpus_name=corpus_name,
        lang=args.lang
    )

    # Preprocess corpus
    dialectifier.preprocess_corpus(args.corpus)

    # Generate corpora for all varieties (incl. standard)
    dialectifier.generate_all_varieties()

    print(f"Done !")
