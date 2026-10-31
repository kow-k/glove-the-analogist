# Benchmark data

The benchmark files are not redistributed here; obtain each from its original
source. `src/batch-evaluate-to-json.sh` looks for them in this folder under the
names below. Set `EVAL_DATA` to use another folder, or set a single variable
(e.g. `ANALOGY=/path/to/file`) to override one file. The wrapper stops if any
file is missing; it does not fall back to the evaluator's small built-in
samples.

| Variable | Default file name | Benchmark | Format read by the evaluator |
|---|---|---|---|
| `STS` | `sts-benchmark.csv` | STS Benchmark (Cer et al., 2017) | `sentence1,sentence2,score`, one pair per line; a header line starting with `sentence` is skipped. Lines are split on commas (tab-separated is accepted when a line has fewer than three comma fields), so commas inside sentences must be removed. |
| `SICK` | `SICK.txt` | SICK (Marelli et al., 2014) | Tab-separated with a header containing `sentence_A`, `sentence_B`, `relatedness_score`. |
| `SIMLEX` | `SimLex-999.txt` | SimLex-999 (Hill et al., 2015) | Tab-separated `word1 word2 score ...`; the header line is skipped. |
| `WORDSIM` | `wordsim353-combined.csv` | WordSim-353 (Finkelstein et al., 2002) | CSV `word1,word2,score` (`combined.csv`). |
| `ANALOGY` | `questions-words.txt` | Google analogy test set (Mikolov et al., 2013) | Original format: `: category` lines, then `a b c d` per line. |
| `OUTLIER` | `outlier-detection.tsv` | 8-8-8 outlier detection (Camacho-Collados and Navigli, 2016) | Either TSV `category<TAB>w1<TAB>...<TAB>outlier` (outlier last), or blocks of 8 lines with the outlier on the 8th. |

STS and SICK pairs with a sentence longer than 12 tokens are skipped
(`--max-input-length`).
