# LegalEval-2023 Task-6C `nclu_team` submissions code

## Installation

![](https://img.shields.io/badge/Python-3.6.9-brightgreen.svg)

```bash
pip install -r dependencies.txt
```

Fist step is to adopt inut processing, provided in:
* `input_orig_convert.py` -- original texts
* `input_v1_convert.py` -- processing, in which duplicate sentences are removed (v1).
* `input_v2_convert.py` -- processing, in which we reduce non semantic oriented sentences (v2).

We provide `cnn` and `att-cnn` models from AREnets.

We provide `RoBERTa_training_notebook.ipynb`, `LegalBERT_training_notebook.ipynb` and `Longformer_training_notebook.ipynb` used for fine-tuning three language models: RoBERTa(Liu et al., 2019), LegalBERT(Chalkidis et al., 2020) and Longformer(Beltagy et al., 2020).


## References
* [AREnets project](https://github.com/nicolay-r/AREnets)
* Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. ArXiv, abs/1907.11692
* Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020. LEGAL-BERT: The muppets straight out of
law school. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2898–2904, Online. Association for Computational Linguistics.
* Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv e-prints, pages arXiv–2004.
