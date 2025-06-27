# LinDiSy

LinDiSy (Linguistically-motivated Dialects Synthesis) aims at creating random language variants using linguistic rules 
to simulate a dialect continuum.

# Pipeline

1) Set phonetic rules manually as in `data/sound_changes/germanic.txt`
2) Build a dialect tree using `preliminary_scripts/make_tree.py` (output = JSON file)
3) Associate phonetic rules to each node of the dialect tree using `preliminary_scripts/generate_tree_rules.py` (output = JSON file)
4) Apply phonetic rules for each dialect on a given corpus using `preliminary_scripts/dialectify.py` (output = 1 TXT file per dialect + 1 for the standard variety + 1 for the preprocessed, phonetized corpus)

# Application on OpenSubtitles

## Corpus preprocessing
The pipeline for preprocessing this corpus includes:
- Downloading the corpus
- Installing and running Bicleaner-AI --> GPU required !
- Filtering results of Bicleaner AI

The English-German OpenSubtitles corpus can be downloaded via [OPUS](https://opus.nlpl.eu/OpenSubtitles/en&de/v2024/OpenSubtitles).
We choose v2024 in Moses format (65M sentence pairs).

In order to clean this corpus using [Bicleaner-AI](https://github.com/bitextor/bicleaner-ai), convert the corpus to TSV (and therefore handle pre-existing tabs):
```bash
cp OpenSubtitles.de-en.en OpenSubtitles.de-en-notabs.en
cp OpenSubtitles.de-en.de OpenSubtitles.de-en-notabs.de
sed -i 's/\t/ /g' OpenSubtitles.de-en-notabs.en
sed -i 's/\t/ /g' OpenSubtitles.de-en-notabs.de
paste OpenSubtitles.de-en-notabs.en OpenSubtitles.de-en-notabs.de > OpenSubtitles.en-de.tsv
```

Install Bicleaner-AI following their documentation:
```bash
pip install bicleaner-ai git+https://github.com/MSeal/cython_hunspell@2.0.3
pip install --config-settings="--build-option=--max_order=7" https://github.com/kpu/kenlm/archive/master.zip
```

(Optionally) Download the "full" version of BiCleaner-AI for English-German (avaiable on [HuggingFace](https://huggingface.co/bitextor/bicleaner-ai-full-en-de)).

Run Bicleaner-AI:
```bash
bicleaner-ai-classify \  
  OpenSubtitles.en-de.tsv \
  OpenSubtitles.en-de.classified.tsv \
  bitextor/bicleaner-ai-full-en-de \
  --scol 1 --tcol 2 \
  -s en -t de \
  --score_only
```

TODO Filter Bicleaner output file with a threshold on the score (goal = ~10M sentences with a very high score)
