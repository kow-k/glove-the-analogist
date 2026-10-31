#!/usr/bin/env python3
"""Minimal word-embedding evaluator (the benchmarks reported in the paper).

A stand-alone reduction of the author's PLM evaluator
(evaluate_embeddings_under_PLM.py) to the plain-embedding path used for every
number in the paper. The Pattern-Lattice-Model (PLM) composition, PPDB,
nearest-neighbour and Jaccard-stratification features are removed; the
benchmark loaders, scoring rules and output keys are kept unchanged, so scores
match the full evaluator on plain word vectors.

Benchmarks and output keys (per-model JSON):
  STS-A   sts_plain    Spearman rho, cosine of averaged word vectors
  SICK-A  sick_plain   (pairs with a sentence longer than --max-input-length
                        tokens are skipped; default 12)
  SL-999  simlex       Spearman rho over in-vocabulary pairs
  WS-353  wordsim      Spearman rho over in-vocabulary pairs
  Analogy analogy      3CosAdd accuracy over quads with all four words in
                        vocabulary; search over the full vocabulary, a/b/c
                        excluded
  Outlier outlier      accuracy of compactness-based outlier detection
  OPP     outlier_opp  Outlier Position Percentage (Camacho-Collados and
                        Navigli 2016)
Any benchmark whose file is not given falls back to a small built-in sample.

Input: word vectors in text format, one "word v1 ... vd" per line, with or
without a leading "<count> <dim>" header (the trainers in this repository write
either, see their --header flag).

Usage:
  python evaluate_embeddings.py model.vec \\
      --sts sts.csv --sick SICK.txt --simlex SimLex-999.txt \\
      --wordsim combined.csv --analogy questions-words.txt \\
      --outlier outlier.tsv --output-json results/{model}.json
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


# =============================================================================
# BUILT-IN BENCHMARK SAMPLES (fallbacks when no benchmark file is given)
# =============================================================================

WORDSIM_353 = [
    ('love', 'sex', 6.77), ('tiger', 'cat', 7.35), ('tiger', 'tiger', 10.0),
    ('book', 'paper', 7.46), ('computer', 'keyboard', 7.62),
    ('computer', 'internet', 7.58), ('plane', 'car', 5.77),
    ('train', 'car', 6.31), ('telephone', 'communication', 7.50),
    ('television', 'radio', 6.77), ('media', 'radio', 7.42),
    ('drug', 'abuse', 6.85), ('bread', 'butter', 6.19),
    ('cucumber', 'potato', 5.92), ('doctor', 'nurse', 7.0),
    ('professor', 'doctor', 6.62), ('student', 'professor', 6.81),
    ('smart', 'student', 4.62), ('smart', 'stupid', 5.81),
    ('company', 'stock', 7.08), ('stock', 'market', 8.08),
    ('stock', 'phone', 1.62), ('stock', 'cd', 1.31),
    ('stock', 'jaguar', 0.92), ('stock', 'egg', 1.08),
    ('fertility', 'egg', 6.69), ('stock', 'live', 3.73),
    ('stock', 'life', 0.92), ('book', 'library', 7.46),
    ('bank', 'money', 8.12), ('wood', 'forest', 7.73),
    ('money', 'cash', 9.15), ('professor', 'cucumber', 0.31),
    ('king', 'queen', 8.58), ('king', 'cabbage', 0.23),
    ('king', 'rook', 5.92), ('bishop', 'rabbi', 6.69),
    ('holy', 'sex', 1.62), ('football', 'basketball', 6.81),
    ('football', 'tennis', 6.63), ('tennis', 'racket', 7.56),
    ('law', 'lawyer', 8.38), ('movie', 'star', 7.38),
    ('movie', 'popcorn', 6.19), ('movie', 'critic', 6.73),
    ('movie', 'theater', 7.92), ('physics', 'chemistry', 7.35),
    ('physics', 'proton', 8.12), ('space', 'chemistry', 4.88),
    ('planet', 'sun', 8.02), ('planet', 'galaxy', 8.11),
    ('planet', 'star', 8.45), ('planet', 'moon', 8.08),
    ('gem', 'jewel', 8.96), ('journey', 'voyage', 9.29),
    ('boy', 'lad', 8.83), ('coast', 'shore', 9.10),
    ('asylum', 'madhouse', 8.87), ('magician', 'wizard', 9.02),
    ('midday', 'noon', 9.29), ('furnace', 'stove', 8.79),
    ('food', 'fruit', 7.52), ('food', 'rooster', 4.42),
    ('bird', 'cock', 7.10), ('bird', 'crane', 7.38),
    ('tool', 'implement', 6.46), ('brother', 'monk', 6.27),
    ('crane', 'implement', 2.37), ('brother', 'lad', 4.46),
    ('car', 'automobile', 8.94), ('forest', 'graveyard', 1.85),
    ('food', 'drink', 6.87), ('coast', 'hill', 4.38),
    ('monk', 'slave', 0.92), ('lad', 'wizard', 0.42),
    ('forest', 'woodland', 8.33), ('cock', 'rooster', 9.44),
    ('boy', 'rooster', 2.54), ('cushion', 'pillow', 8.00),
    ('cemetery', 'woodland', 2.08), ('automobile', 'wizard', 0.21),
    ('grin', 'smile', 8.49), ('serf', 'slave', 8.54),
    ('shore', 'woodland', 3.08), ('glass', 'magician', 2.08),
    ('rooster', 'voyage', 0.62), ('noon', 'string', 0.54),
    ('glass', 'jewel', 1.96), ('magician', 'oracle', 5.46),
    ('brother', 'sister', 8.27), ('sage', 'wizard', 6.62),
    ('hill', 'mound', 7.79), ('car', 'journey', 5.85),
    ('cemetery', 'mound', 5.42), ('glass', 'tumbler', 7.25),
    ('smile', 'village', 0.58), ('pond', 'lake', 7.73),
    ('car', 'flight', 4.94), ('bird', 'woodland', 5.33),
    ('emergency', 'hospital', 7.73), ('stock', 'investment', 7.92),
    ('fruit', 'vegetable', 6.67),
]

SIMLEX_999 = [
    ('old', 'new', 1.58), ('smart', 'intelligent', 9.20),
    ('hard', 'difficult', 8.77), ('happy', 'cheerful', 9.55),
    ('hard', 'easy', 0.95), ('fast', 'rapid', 8.75),
    ('happy', 'glad', 9.17), ('short', 'long', 1.23),
    ('stupid', 'dumb', 9.58), ('weird', 'strange', 8.93),
    ('wide', 'narrow', 1.03), ('bad', 'good', 0.65),
    ('mad', 'angry', 8.05), ('young', 'old', 0.98),
    ('big', 'large', 8.47), ('big', 'small', 0.82),
    ('good', 'great', 6.83), ('bad', 'terrible', 7.78),
    ('hot', 'cold', 0.77), ('cheap', 'expensive', 0.78),
    ('strong', 'powerful', 8.22), ('weak', 'strong', 1.38),
    ('thin', 'thick', 1.27), ('small', 'little', 8.75),
    ('fast', 'slow', 0.62), ('bright', 'smart', 7.22),
    ('tall', 'short', 0.70), ('clean', 'dirty', 1.43),
    ('rich', 'poor', 1.17), ('pretty', 'beautiful', 8.08),
    ('empty', 'full', 0.90), ('nice', 'mean', 1.13),
    ('huge', 'tiny', 0.72), ('easy', 'difficult', 0.58),
    ('dark', 'bright', 1.37), ('alive', 'dead', 0.55),
    ('woman', 'man', 3.33), ('king', 'queen', 5.07),
    ('brother', 'sister', 4.77), ('car', 'automobile', 9.22),
    ('cup', 'mug', 8.08), ('baby', 'infant', 9.53),
    ('movie', 'film', 9.47), ('river', 'stream', 7.77),
    ('dog', 'cat', 5.20), ('doctor', 'nurse', 5.00),
    ('house', 'home', 8.10), ('chair', 'table', 4.87),
    ('car', 'bus', 5.50), ('bird', 'eagle', 7.53),
    ('country', 'nation', 8.63), ('city', 'town', 7.62),
    ('ship', 'boat', 7.47), ('apple', 'orange', 5.63),
    ('hat', 'cap', 7.17), ('music', 'song', 7.27),
    ('road', 'street', 8.10), ('phone', 'telephone', 9.75),
    ('ocean', 'sea', 9.08), ('mountain', 'hill', 7.45),
    ('bread', 'cake', 5.30), ('student', 'pupil', 9.07),
    ('friend', 'enemy', 2.18), ('work', 'job', 8.17),
    ('mother', 'father', 5.17), ('wife', 'husband', 6.08),
    ('sun', 'moon', 4.02), ('earth', 'planet', 8.00),
    ('child', 'kid', 9.30), ('money', 'cash', 9.12),
    ('love', 'hate', 2.17), ('laugh', 'cry', 2.10),
    ('buy', 'sell', 2.33), ('begin', 'end', 2.13),
    ('huge', 'enormous', 9.00), ('fast', 'quick', 8.65),
    ('smart', 'clever', 8.95),
]

BUILTIN_STS = [
    ("A man is playing a guitar.", "A man is playing music.", 4.2),
    ("A woman is cooking dinner.", "A woman is preparing a meal.", 4.5),
    ("The dog is running in the park.", "A dog runs through a park.", 4.3),
    ("A child is reading a book.", "A kid reads a book.", 4.8),
    ("The cat sleeps on the couch.", "A cat is sleeping on a sofa.", 4.6),
    ("A man is riding a bicycle.", "A person rides a bike.", 4.4),
    ("The sun is shining brightly.", "The sun shines bright.", 4.7),
    ("A woman is singing a song.", "A woman sings.", 4.1),
    ("The bird flies in the sky.", "A bird is flying.", 4.0),
    ("A man is swimming in the pool.", "A man swims in a swimming pool.", 4.5),
    ("A man is playing a piano.", "A man is playing a guitar.", 2.8),
    ("A woman is walking her dog.", "A woman is running with her cat.", 2.2),
    ("The children are playing soccer.", "The kids are watching television.", 1.8),
    ("A man is eating an apple.", "A man is drinking coffee.", 2.0),
    ("The students are studying math.", "The children are learning science.", 3.0),
    ("A dog is chasing a ball.", "A cat is playing with yarn.", 2.4),
    ("A man is writing a letter.", "A woman is typing an email.", 3.2),
    ("A man is playing basketball.", "A woman is cooking dinner.", 0.5),
    ("The sun is setting.", "The children are sleeping.", 0.8),
    ("A dog is barking.", "A fish is swimming.", 1.0),
    ("A woman is dancing.", "A man is sleeping.", 0.3),
    ("A child is crying.", "A bird is singing.", 0.9),
    ("A man is running fast.", "A woman is sitting quietly.", 0.7),
    ("The book is interesting.", "The weather is cold.", 0.2),
    ("Two men are fighting.", "Two men are wrestling.", 3.8),
    ("A woman is slicing vegetables.", "A woman is cutting tomatoes.", 3.5),
    ("A man is lifting weights.", "A person is exercising.", 3.7),
    ("A woman is painting a picture.", "An artist is drawing.", 3.4),
    ("The snow is falling.", "Rain is pouring down.", 2.5),
    ("The student passed the exam.", "The learner succeeded in the test.", 4.5),
]

BUILTIN_SICK = [
    ("A man is playing a guitar", "A man is playing a musical instrument", 4.5),
    ("A woman is cutting vegetables", "A woman is slicing vegetables", 4.8),
    ("A dog is running through the grass", "A dog is running in a field", 4.3),
    ("A child is playing with a toy", "A kid is playing with toys", 4.6),
    ("A man is riding a horse", "A person is on horseback", 4.4),
    ("Two people are dancing", "A couple is dancing together", 4.2),
    ("A cat is sitting on a mat", "A cat is resting on a rug", 4.1),
    ("A man is swimming in the ocean", "A man is swimming in the sea", 4.9),
    ("A man is playing a guitar", "A man is playing a piano", 3.2),
    ("A woman is cooking", "A woman is eating", 2.8),
    ("A dog is running", "A cat is sleeping", 1.8),
    ("A child is reading", "A child is writing", 2.5),
    ("A man is walking", "A man is running", 3.0),
    ("A bird is flying", "A bird is sitting", 2.4),
    ("A car is moving", "A car is parked", 2.2),
    ("A man is playing guitar", "A woman is cooking dinner", 1.2),
    ("A dog is barking", "A fish is swimming", 1.0),
    ("A child is sleeping", "A car is driving", 1.1),
    ("A woman is running", "A man is eating", 1.3),
    ("A cat is meowing", "A plane is flying", 1.0),
    ("A man is standing", "Nobody is standing", 1.2),
    ("The dog is white", "The dog is black", 2.0),
    ("A child is happy", "A child is sad", 1.7),
    ("The room is empty", "The room is full", 1.5),
    ("Someone is frying food", "Someone is cooking food", 4.3),
    ("A man is slicing bread", "A man is cutting bread", 4.8),
    ("A woman is mixing ingredients", "A woman is stirring ingredients", 4.6),
    ("A child is drawing a picture", "A kid is drawing a picture", 4.7),
    ("A dog is chasing a ball", "A dog is pursuing a ball", 4.6),
    ("A cat is climbing a tree", "A cat is scaling a tree", 4.7),
]

GOOGLE_ANALOGY = [
    # Semantic: capital-country
    ('athens', 'greece', 'baghdad', 'iraq'),
    ('athens', 'greece', 'bangkok', 'thailand'),
    ('athens', 'greece', 'beijing', 'china'),
    ('berlin', 'germany', 'bern', 'switzerland'),
    ('berlin', 'germany', 'cairo', 'egypt'),
    ('london', 'england', 'paris', 'france'),
    ('paris', 'france', 'rome', 'italy'),
    ('tokyo', 'japan', 'moscow', 'russia'),
    ('madrid', 'spain', 'canberra', 'australia'),
    ('ottawa', 'canada', 'london', 'england'),
    # Semantic: capital-world
    ('abuja', 'nigeria', 'accra', 'ghana'),
    ('algiers', 'algeria', 'ankara', 'turkey'),
    ('astana', 'kazakhstan', 'athens', 'greece'),
    ('berlin', 'germany', 'tokyo', 'japan'),
    ('hanoi', 'vietnam', 'havana', 'cuba'),
    # Semantic: currency
    ('algeria', 'dinar', 'angola', 'kwanza'),
    ('argentina', 'peso', 'australia', 'dollar'),
    ('brazil', 'real', 'canada', 'dollar'),
    ('europe', 'euro', 'usa', 'dollar'),
    ('japan', 'yen', 'uk', 'pound'),
    ('india', 'rupee', 'iran', 'rial'),
    # Semantic: city-in-state
    ('chicago', 'illinois', 'houston', 'texas'),
    ('dallas', 'texas', 'boston', 'massachusetts'),
    ('denver', 'colorado', 'detroit', 'michigan'),
    ('miami', 'florida', 'phoenix', 'arizona'),
    ('seattle', 'washington', 'atlanta', 'georgia'),
    # Semantic: family
    ('boy', 'girl', 'brother', 'sister'),
    ('boy', 'girl', 'father', 'mother'),
    ('boy', 'girl', 'husband', 'wife'),
    ('boy', 'girl', 'king', 'queen'),
    ('boy', 'girl', 'man', 'woman'),
    ('brother', 'sister', 'grandson', 'granddaughter'),
    ('brother', 'sister', 'king', 'queen'),
    ('father', 'mother', 'uncle', 'aunt'),
    ('husband', 'wife', 'king', 'queen'),
    ('king', 'queen', 'man', 'woman'),
    ('man', 'woman', 'prince', 'princess'),
    ('nephew', 'niece', 'son', 'daughter'),
    # Syntactic: adjective-to-adverb
    ('amazing', 'amazingly', 'apparent', 'apparently'),
    ('calm', 'calmly', 'quick', 'quickly'),
    ('happy', 'happily', 'lucky', 'luckily'),
    ('obvious', 'obviously', 'rapid', 'rapidly'),
    ('slow', 'slowly', 'sudden', 'suddenly'),
    # Syntactic: opposite
    ('aware', 'unaware', 'certain', 'uncertain'),
    ('clear', 'unclear', 'comfortable', 'uncomfortable'),
    ('happy', 'unhappy', 'likely', 'unlikely'),
    ('possible', 'impossible', 'rational', 'irrational'),
    ('acceptable', 'unacceptable', 'aware', 'unaware'),
    # Syntactic: comparative
    ('bad', 'worse', 'big', 'bigger'),
    ('big', 'bigger', 'bright', 'brighter'),
    ('cold', 'colder', 'cool', 'cooler'),
    ('fast', 'faster', 'good', 'better'),
    ('great', 'greater', 'high', 'higher'),
    ('large', 'larger', 'long', 'longer'),
    ('low', 'lower', 'old', 'older'),
    ('sharp', 'sharper', 'short', 'shorter'),
    ('simple', 'simpler', 'slow', 'slower'),
    ('smart', 'smarter', 'strong', 'stronger'),
    # Syntactic: superlative
    ('bad', 'worst', 'big', 'biggest'),
    ('bright', 'brightest', 'cool', 'coolest'),
    ('fast', 'fastest', 'good', 'best'),
    ('great', 'greatest', 'high', 'highest'),
    ('large', 'largest', 'long', 'longest'),
    ('old', 'oldest', 'sharp', 'sharpest'),
    ('simple', 'simplest', 'slow', 'slowest'),
    ('smart', 'smartest', 'strong', 'strongest'),
    ('tall', 'tallest', 'warm', 'warmest'),
    ('young', 'youngest', 'wide', 'widest'),
    # Syntactic: present participle (verb -ing)
    ('dance', 'dancing', 'describe', 'describing'),
    ('fly', 'flying', 'go', 'going'),
    ('look', 'looking', 'move', 'moving'),
    ('play', 'playing', 'run', 'running'),
    ('say', 'saying', 'sing', 'singing'),
    ('walk', 'walking', 'write', 'writing'),
    # Syntactic: nationality-adjective
    ('albania', 'albanian', 'australia', 'australian'),
    ('brazil', 'brazilian', 'cambodia', 'cambodian'),
    ('china', 'chinese', 'france', 'french'),
    ('germany', 'german', 'india', 'indian'),
    ('israel', 'israeli', 'japan', 'japanese'),
    ('korea', 'korean', 'switzerland', 'swiss'),
    # Syntactic: past tense
    ('dancing', 'danced', 'describing', 'described'),
    ('flying', 'flew', 'going', 'went'),
    ('looking', 'looked', 'moving', 'moved'),
    ('playing', 'played', 'running', 'ran'),
    ('saying', 'said', 'singing', 'sang'),
    ('walking', 'walked', 'writing', 'wrote'),
    # Syntactic: plural
    ('banana', 'bananas', 'bird', 'birds'),
    ('car', 'cars', 'child', 'children'),
    ('dollar', 'dollars', 'dog', 'dogs'),
    ('eye', 'eyes', 'finger', 'fingers'),
    ('horse', 'horses', 'man', 'men'),
    ('mouse', 'mice', 'woman', 'women'),
    # Syntactic: plural verbs
    ('decrease', 'decreases', 'describe', 'describes'),
    ('eat', 'eats', 'find', 'finds'),
    ('go', 'goes', 'increase', 'increases'),
    ('play', 'plays', 'run', 'runs'),
    ('say', 'says', 'see', 'sees'),
    ('think', 'thinks', 'walk', 'walks'),
]

BUILTIN_OUTLIER = [
    # Animals
    ("animals_farm",      ["cow","pig","sheep","goat","horse","chicken","duck"],        "eagle"),
    ("animals_wild",      ["lion","tiger","elephant","bear","wolf","deer","fox"],       "salmon"),
    ("animals_pets",      ["dog","cat","rabbit","bird","fish","mouse","rat"],           "wolf"),
    ("animals_birds",     ["eagle","hawk","owl","duck","crow","robin","swan"],          "shark"),
    # Fruits and vegetables
    ("fruits",            ["apple","banana","orange","peach","cherry","grape","lemon"], "potato"),
    ("vegetables",        ["carrot","potato","onion","tomato","pepper","bean","corn"],  "apple"),
    # Countries
    ("european_countries",["france","germany","italy","spain","russia","poland","sweden"],"brazil"),
    ("asian_countries",   ["japan","china","korea","india","iran","iraq","turkey"],     "kenya"),
    # Sports
    ("team_sports",       ["football","basketball","baseball","hockey","rugby","tennis","cricket"],"swimming"),
    ("water_sports",      ["swimming","diving","sailing","fishing","rowing","surfing","racing"],"tennis"),
    # Occupations
    ("medical",           ["doctor","nurse","dentist","teacher","lawyer","scientist","engineer"],"soldier"),
    ("legal",             ["lawyer","judge","police","court","crime","justice","law"],  "teacher"),
    # Body parts
    ("facial",            ["eye","nose","mouth","ear","chin","cheek","hair"],           "knee"),
    ("limbs",             ["arm","leg","hand","foot","knee","finger","toe"],            "head"),
    # Colors
    ("warm_colors",       ["red","orange","yellow","pink","brown","gold","white"],      "blue"),
    ("cool_colors",       ["blue","green","purple","black","gray","silver","white"],    "red"),
    # Furniture
    ("seating",           ["chair","sofa","bench","stool","bed","desk","table"],        "lamp"),
    ("storage",           ["cabinet","shelf","box","bag","drawer","case","container"],  "lamp"),
    # Musical instruments
    ("strings",           ["guitar","violin","piano","drum","bass","cello","harp"],     "trumpet"),
    ("wind",              ["trumpet","flute","clarinet","saxophone","horn","pipe","bell"],"guitar"),
    # Science
    ("elements_metals",   ["iron","copper","gold","silver","steel","stone","wood"],     "water"),
    ("planets",           ["mercury","venus","mars","jupiter","saturn","earth","moon"], "sun"),
    # Transportation
    ("land_vehicles",     ["car","bus","truck","train","bike","ship","plane"],          "boat"),
    ("water_vehicles",    ["boat","ship","vessel","craft","fleet","navy","port"],       "car"),
    # Emotions
    ("positive_emotions", ["joy","happiness","love","hope","pride","peace","faith"],    "anger"),
    ("negative_emotions", ["anger","fear","pain","hate","grief","shame","guilt"],       "joy"),
    # Foods
    ("grains",            ["wheat","rice","corn","bread","flour","seed","grain"],       "meat"),
    ("meats",             ["beef","pork","chicken","lamb","turkey","fish","egg"],       "bread"),
    # Academic subjects
    ("sciences",          ["physics","chemistry","biology","astronomy","medicine","mathematics","engineering"],"history"),
    ("humanities",        ["history","philosophy","literature","religion","art","music","culture"],"physics"),
]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_vectors(path):
    """Load text-format word vectors (header optional).

    Returns a model dict with 'word2idx' and 'vectors' (float32, V x d).
    Lines whose dimensionality differs from the first vector line are skipped.
    """
    words, rows = [], []
    dim = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n").split()
        if not (len(first) == 2 and all(t.isdigit() for t in first)):
            f.seek(0)                      # no header: rewind and read line 1
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 2:
                continue
            if dim is None:
                dim = len(parts) - 1
            if len(parts) - 1 != dim:
                continue
            try:
                rows.append(np.asarray(parts[1:], dtype=np.float32))
            except ValueError:
                continue
            words.append(parts[0])
    if not rows:
        raise ValueError(f"no vectors read from {path}")
    w2i = {}
    for i, w in enumerate(words):
        w2i[w] = i                         # a duplicated word keeps its last row
    return {"word2idx": w2i, "vectors": np.vstack(rows)}


def word_vector(word, model):
    idx = model["word2idx"].get(word)
    return None if idx is None else model["vectors"][idx]


def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return np.dot(v1, v2) / (n1 * n2)


# =============================================================================
# BENCHMARK LOADERS (formats as in the full evaluator)
# =============================================================================

def load_wordsim353(filepath):
    """WordSim-353 CSV: either the 'en_word_1' multi-column form or word1,word2,score."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and "en_word_1" in header:
            for row in reader:
                if len(row) > 12:
                    try:
                        pairs.append((row[3].strip().lower(), row[4].strip().lower(),
                                      float(row[12].strip())))
                    except (ValueError, IndexError):
                        continue
        else:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line or line.startswith("Word"):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
                    except ValueError:
                        continue
    return pairs


def load_simlex999(filepath):
    """SimLex-999 (tab- or space-separated; header line 'word1 ...' skipped)."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("word1"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) >= 3:
                try:
                    pairs.append((parts[0].strip().lower(), parts[1].strip().lower(),
                                  float(parts[2])))
                except (ValueError, IndexError):
                    continue
    return pairs


def load_sts_file(filepath):
    """STS: sentence1,sentence2,score (comma-separated, or tab-separated)."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("sentence"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    pairs.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))
                except ValueError:
                    continue
    return pairs


def load_sick_file(filepath):
    """SICK: tab-separated with sentence_A / sentence_B / relatedness_score columns."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        header_parts = f.readline().strip().split("\t")
        try:
            a = header_parts.index("sentence_A")
            b = header_parts.index("sentence_B")
            s = header_parts.index("relatedness_score")
        except ValueError:
            a, b, s = 1, 2, 3
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(a, b, s):
                try:
                    pairs.append((parts[a].strip(), parts[b].strip(), float(parts[s].strip())))
                except (ValueError, IndexError):
                    continue
    return pairs


def load_analogy_file(filepath):
    """Google analogy format (': category' headers, then 'a b c d' lines)."""
    analogies, cats = [], []
    current = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                current = line[1:].strip()
            else:
                parts = line.lower().split()
                if len(parts) == 4:
                    analogies.append(tuple(parts))
                    cats.append(current)
    return analogies, cats


def load_outlier_file(filepath):
    """Outlier sets. TSV: category<TAB>w1<TAB>...<TAB>outlier (outlier last);
    or blocks of 8 lines, the 8th line being the outlier."""
    clusters = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not lines:
        return clusters
    if "\t" in lines[0]:
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 4:
                clusters.append((parts[0], parts[1:-1], parts[-1]))
        return clusters
    for i in range(0, len(lines) - 7, 8):
        block = lines[i:i + 8]
        if len(block) == 8:
            clusters.append((f"cluster_{i // 8}", block[:7], block[7]))
    return clusters


# =============================================================================
# SCORING
# =============================================================================

def sentence_vector_plain(sentence, model):
    """Average of in-vocabulary word vectors (lower-cased, basic punctuation stripped)."""
    words = sentence.lower().replace(".", "").replace(",", "").replace(
        "!", "").replace("?", "").replace(";", "").replace(":", "").split()
    vecs = [model["vectors"][model["word2idx"][w]] for w in words if w in model["word2idx"]]
    return np.mean(vecs, axis=0) if vecs else None


def evaluate_sentence_similarity(model, pairs, max_input_length=12):
    """Spearman rho for averaged-vector cosine; returns (rho, n_scored, n_total, n_skipped)."""
    gold, pred = [], []
    n_skipped = 0
    for s1, s2, score in pairs:
        if max_input_length is not None and (
                len(s1.split()) > max_input_length or len(s2.split()) > max_input_length):
            n_skipped += 1
            continue
        v1, v2 = sentence_vector_plain(s1, model), sentence_vector_plain(s2, model)
        if v1 is not None and v2 is not None:
            pred.append(cosine_similarity(v1, v2))
            gold.append(score)
    rho = spearmanr(gold, pred)[0] if len(pred) >= 3 else None
    return rho, len(pred), len(pairs), n_skipped


def evaluate_word_similarity(model, pairs):
    """Spearman rho over pairs with both words in vocabulary; (rho, n_found, n_total)."""
    gold, pred = [], []
    for w1, w2, score in pairs:
        v1, v2 = word_vector(w1.lower(), model), word_vector(w2.lower(), model)
        if v1 is not None and v2 is not None:
            gold.append(score)
            pred.append(cosine_similarity(v1, v2))
    rho = spearmanr(gold, pred)[0] if len(gold) >= 3 else None
    return rho, len(gold), len(pairs)


def evaluate_analogy(model, analogies, categories=None, sample_ratio=None, seed=42):
    """3CosAdd: predicted = argmax_w cos(w, b - a + c), w not in {a, b, c}.

    Quads with any out-of-vocabulary word are skipped; accuracy is over the rest.
    """
    if categories is not None and len(categories) == len(analogies):
        paired = list(zip(analogies, categories))
    else:
        paired = [(q, None) for q in analogies]
    order = []
    for _, c in paired:
        if c is not None and c not in order:
            order.append(c)
    if sample_ratio is not None and 0 < sample_ratio < 1:
        random.seed(seed)
        paired = random.sample(paired, max(1, int(len(paired) * sample_ratio)))

    w2i, E = model["word2idx"], model["vectors"]
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1
    En = E / norms

    correct = total = skipped = 0
    c_ok, c_tot, c_skip = {}, {}, {}
    for (a, b, c, d), cat in paired:
        va, vb, vc, vd = (word_vector(w, model) for w in (a, b, c, d))
        if va is None or vb is None or vc is None or vd is None:
            skipped += 1
            if cat is not None:
                c_skip[cat] = c_skip.get(cat, 0) + 1
            continue
        target = vb - va + vc
        tn = np.linalg.norm(target)
        if tn == 0:
            skipped += 1
            if cat is not None:
                c_skip[cat] = c_skip.get(cat, 0) + 1
            continue
        sims = En @ (target / tn)
        for w in (a, b, c):
            if w in w2i:
                sims[w2i[w]] = -2.0
        if int(np.argmax(sims)) == w2i[d]:
            correct += 1
            if cat is not None:
                c_ok[cat] = c_ok.get(cat, 0) + 1
        total += 1
        if cat is not None:
            c_tot[cat] = c_tot.get(cat, 0) + 1

    per_category = {}
    for cat in order:
        ct, sk = c_tot.get(cat, 0), c_skip.get(cat, 0)
        per_category[cat] = {
            "accuracy": (c_ok.get(cat, 0) / ct) if ct else None,
            "correct": c_ok.get(cat, 0), "total": ct, "skipped": sk,
            "coverage": ct / (ct + sk) if (ct + sk) else 0.0,
        }
    return {
        "accuracy": correct / total if total else None,
        "correct": correct, "total": total, "skipped": skipped,
        "sampled": len(paired),
        "coverage": total / (total + skipped) if (total + skipped) else 0,
        "per_category": per_category,
    }


def evaluate_outlier_detection(model, clusters):
    """Compactness scoring: a word's compactness is its mean cosine to the other
    words of its set; the least compact word is predicted as the outlier.
    Sets with any out-of-vocabulary word are skipped.

    OPP = 1 - mean(rank of true outlier / (N - 1)), rank 0 = least compact.
    """
    correct = tested = skipped = 0
    opp_sum = 0.0
    for _, members, outlier in clusters:
        all_words = members + [outlier]
        vecs = {}
        for w in all_words:
            v = word_vector(w, model)
            if v is not None:
                vecs[w] = v
        if len(vecs) < len(all_words):        # OOV word (or a repeated word)
            skipped += 1
            continue
        comp = {w: float(np.mean([cosine_similarity(vecs[w], vecs[o])
                                  for o in all_words if o != w]))
                for w in all_words}
        ranked = sorted(all_words, key=lambda w: comp[w])
        opp_sum += ranked.index(outlier) / (len(all_words) - 1)
        if ranked[0] == outlier:
            correct += 1
        tested += 1
    return {
        "accuracy": correct / tested if tested else None,
        "opp": 1.0 - (opp_sum / tested) if tested else None,
        "correct": correct, "tested": tested, "skipped": skipped, "total": len(clusters),
    }


# =============================================================================
# DRIVER
# =============================================================================

def _fmt(x):
    return "N/A" if x is None else f"{x:.4f}"


def evaluate_model(path, args):
    name = Path(path).name
    model = load_vectors(path)
    V, d = model["vectors"].shape
    print(f"\n{'=' * 70}\nEvaluating: {name}   ({V} words, d={d})\n{'=' * 70}")
    r = {"name": name}

    sts = load_sts_file(args.sts) if args.sts and os.path.exists(args.sts) else BUILTIN_STS
    rho, n, tot, sk = evaluate_sentence_similarity(model, sts, args.max_input_length)
    r.update(sts_plain=rho, sts_coverage=n, sts_total=tot)
    print(f"  STS-A:    {_fmt(rho)}  ({n}/{tot} pairs, {sk} over length)")

    sick = load_sick_file(args.sick) if args.sick and os.path.exists(args.sick) else BUILTIN_SICK
    rho, n, tot, sk = evaluate_sentence_similarity(model, sick, args.max_input_length)
    r.update(sick_plain=rho, sick_coverage=n, sick_total=tot)
    print(f"  SICK-A:   {_fmt(rho)}  ({n}/{tot} pairs, {sk} over length)")

    sl = load_simlex999(args.simlex) if args.simlex else SIMLEX_999
    rho, n, tot = evaluate_word_similarity(model, sl)
    r.update(simlex=rho, simlex_coverage=n, simlex_total=tot)
    print(f"  SL-999:   {_fmt(rho)}  ({n}/{tot} pairs)")

    ws = load_wordsim353(args.wordsim) if args.wordsim else WORDSIM_353
    rho, n, tot = evaluate_word_similarity(model, ws)
    r.update(wordsim=rho, wordsim_coverage=n, wordsim_total=tot)
    print(f"  WS-353:   {_fmt(rho)}  ({n}/{tot} pairs)")

    if args.analogy and os.path.exists(args.analogy):
        quads, cats = load_analogy_file(args.analogy)
    else:
        quads, cats = GOOGLE_ANALOGY, None
    an = evaluate_analogy(model, quads, cats, sample_ratio=args.analogy_sample)
    r.update(analogy=an["accuracy"], analogy_correct=an["correct"],
             analogy_total=an["total"], analogy_skipped=an["skipped"],
             analogy_sampled=an["sampled"], analogy_per_category=an["per_category"])
    print(f"  Analogy:  {_fmt(an['accuracy'])}  ({an['correct']}/{an['total']}, "
          f"{an['skipped']} skipped)")

    cl = (load_outlier_file(args.outlier) if args.outlier and os.path.exists(args.outlier)
          else BUILTIN_OUTLIER)
    od = evaluate_outlier_detection(model, cl)
    r.update(outlier=od["accuracy"], outlier_opp=od["opp"], outlier_correct=od["correct"],
             outlier_tested=od["tested"], outlier_skipped=od["skipped"])
    print(f"  Outlier:  {_fmt(od['accuracy'])}  OPP {_fmt(od['opp'])}  "
          f"({od['tested']} sets tested, {od['skipped']} skipped)")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="word-vector file(s), text format")
    ap.add_argument("--sts", help="STS file (sentence1,sentence2,score)")
    ap.add_argument("--sick", help="SICK file (tab-separated, with header)")
    ap.add_argument("--simlex", help="SimLex-999 file")
    ap.add_argument("--wordsim", help="WordSim-353 CSV")
    ap.add_argument("--analogy", help="analogy file (questions-words.txt format)")
    ap.add_argument("--analogy-sample", type=float, default=None,
                    help="evaluate a random fraction of the analogy quads (seed 42)")
    ap.add_argument("--outlier", help="outlier-detection file (TSV or 8-line blocks)")
    ap.add_argument("--max-input-length", type=int, default=12,
                    help="skip STS/SICK pairs with a sentence longer than this (default 12)")
    ap.add_argument("--output-json", default=None,
                    help="write per-model JSON: a path containing {model} (replaced by the "
                         "file stem) or a directory")
    a = ap.parse_args()

    for path in sorted(a.models):
        try:
            r = evaluate_model(path, a)
        except Exception as e:                        # keep going over a batch
            print(f"  ERROR evaluating {path}: {e}", file=sys.stderr)
            continue
        if a.output_json:
            stem = Path(path).stem
            jp = (a.output_json.replace("{model}", stem) if "{model}" in a.output_json
                  else os.path.join(a.output_json, stem + ".json"))
            os.makedirs(os.path.dirname(jp) or ".", exist_ok=True)
            with open(jp, "w") as jf:
                json.dump(r, jf, indent=2, default=float)
            print(f"  Saved: {jp}")


if __name__ == "__main__":
    main()
