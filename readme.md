# [NevIR: Negation in Neural Information Retrieval](https://arxiv.org/abs/2305.07614)
By Orion Weller, Dawn Lawrie, and Benjamin Van Durme

## Quick Links:
- Paper: https://arxiv.org/abs/2305.07614
- Dataset on Huggingface: https://huggingface.co/datasets/orionweller/NevIR

## How to evaluate models on NevIR
0. Install the dependencies in the requirement files (`./bin/install`) after installing a conda enviroment from `conda_env.yml` (e.g. `conda env create -f conda_env.yml`). See "Extra Installation" section for ColBERT and MonoT5 installations or "Install All" to install all models
1. Choose the models you'd like to evaluate by modifying the code in `src/negations/evaluate.py`. You can easily add new models from sentence-transformers or huggingface (e.g. add dense models following the examples by replacing the `model_name`)
2. Run `python src/negations/evaluate.py` to evaluate. This is much faster if you have a GPU. See run options in the file (save output, etc.)


## Extra Installation for other models
- For ColBERT download the github repo: `git clone https://github.com/stanford-futuredata/ColBERT.git && cd ColBERT && git checkout colbertv1 && pip install .`.  You'll also need to create a directory called `colbert_weights` and place the weights from colbertv1 inside (can take weights from https://huggingface.co/orionweller/ColBERTv1/tree/main). To use v2, you'll need to modify the original ColBERT code to take in the new Checkpoint type.
- For MonoT5 install `pip install git+https://github.com/castorini/pygaggle`
- For Splade install `pip install git+https://github.com/naver/splade.git`

## Install All
0. You can follow the ColBERT repo instructions above and install from the `all_conda_env.yml` file instead. This has all the dependencies.


## Training models on NevIR
Due to current deadlines, I will upload this code if someone is interested in it, it is just a training script from sentence-transformers/original ColBERT repo with little modifications.

You can easily incorporate it into your framework of choice by using the examples as training triples (since it is composed of positive and negative pairs).


## Citation
If you used NevIR in your work, please consider citing:
```
@misc{weller2023nevir,
      title={NevIR: Negation in Neural Information Retrieval}, 
      author={Orion Weller and Dawn Lawrie and Benjamin Van Durme},
      year={2023},
      eprint={2305.07614},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```