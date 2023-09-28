import os
import json
import pandas as pd
import argparse
import ast
import numpy as np
import tqdm
from collections import defaultdict
from scipy.special import softmax
import torch
from datasets import load_dataset

# NOTE: if you are adding a new type of model not covered by these 
# frameworks you'll want to add and import it here
from negations.dense import calc_preferred_dense
from negations.tfidf import calc_preferred_tfidf
from negations.rocketqa_models import calc_preferred_rocketqa

# add these with special installs
# from negations.colbert import calc_preferred_colbert 
from negations.splade_models import calc_preferred_splade
from negations.rerank_pygaggle import calc_preferred_rerankers

# TODO: include the models you want to use
# the format is (pretty_model_name, model_architecture_func, model_details)

MODEL_MAP = {
    "tfidf": ("tfidf", calc_preferred_tfidf, None),
    # "colbertv1": ("colbertv1", calc_preferred_colbert, "v1"),
    # "colbertv2": ("colbertv2", calc_preferred_colbert, "v2"),
    # "monoT5-large": ("monoT5-large", calc_preferred_rerankers, "castorini/monot5-large-msmarco-10k"),
    # "monoT5-3b": ("monoT5-3b", calc_preferred_rerankers, "castorini/monot5-3b-msmarco-10k"),
    # "monoT5-base": ("monoT5-base", calc_preferred_rerankers, "castorini/monot5-base-msmarco-10k"),
    "monoT5-small": ("monoT5-small", calc_preferred_rerankers, "castorini/monot5-small-msmarco-10k"),
    # "cross-encoder/ms-marco-MiniLM-L-12-v2": (
    #     "cross-encoder-ms-marco-MiniLM-L-12-v2",
    #     calc_preferred_dense,
    #     "cross-encoder/ms-marco-MiniLM-L-12-v2",
    # ),
    # "cross-encoder/qnli-electra-base": (
    #     "cross-encoder-qnli-electra-base",
    #     calc_preferred_dense,
    #     "cross-encoder/qnli-electra-base",
    # ),
    # "cross-encoder/stsb-roberta-large": (
    #     "cross-encoder-stsb-roberta-large",
    #     calc_preferred_dense,
    #     "cross-encoder/stsb-roberta-large",
    # ),
    # "cross-encoder/nli-deberta-v3-base": (
    #     "cross-encoder-nli-deberta-v3-base",
    #     calc_preferred_dense,
    #     "cross-encoder/nli-deberta-v3-base",
    # ),
    "msmarco-bert-base-dot-v5": (
        "msmarco-bert-base-dot-v5",
        calc_preferred_dense,
        "msmarco-bert-base-dot-v5",
    ),
    # "msmarco-distilbert-cos-v5": (
    #     "msmarco-distilbert-cos-v5",
    #     calc_preferred_dense,
    #     "msmarco-distilbert-cos-v5",
    # ),
    # "all-mpnet-base-v2": (
    #     "all-mpnet-base-v2",
    #     calc_preferred_dense,
    #     "all-mpnet-base-v2",
    # ),
    # "multi-qa-mpnet-base-dot-v1": (
    #     "multi-qa-mpnet-base-dot-v1",
    #     calc_preferred_dense,
    #     "multi-qa-mpnet-base-dot-v1",
    # ),
    # "nq-distilbert-base-v1": (
    #     "nq-distilbert-base-v1",
    #     calc_preferred_dense,
    #     "nq-distilbert-base-v1",
    # ),
    # "dpr": (
    #     "dpr",
    #     calc_preferred_dense,
    #     "dpr",
    # ),
    # "msmarco-bert-co-condensor": (
    #     "msmarco-bert-co-condensor",
    #     calc_preferred_dense,
    #     "msmarco-bert-co-condensor",
    # ),
    # "v2_marco_ce": (
    #     "v2_marco_ce",
    #     calc_preferred_rocketqa,
    #     "v2_marco_ce",
    # ),
    # "v2_marco_de": (
    #     "v2_marco_de",
    #     calc_preferred_rocketqa,
    #     "v2_marco_de",
    # ),
    # "v1_marco_ce": (
    #     "v1_marco_ce",
    #     calc_preferred_rocketqa,
    #     "v1_marco_ce",
    # ),
    "v1_marco_de": (
        "v1_marco_de",
        calc_preferred_rocketqa,
        "v1_marco_de",
    ),
    "naver/splade-cocondenser-selfdistil": (
        "splade-cocondenser-selfdistil",
        calc_preferred_splade,
        "naver/splade-cocondenser-selfdistil",
    ),
    # "naver/splade-cocondenser-ensembledistil": (
    #     "splade-cocondenser-ensembledistil",
    #     calc_preferred_splade,
    #     "naver/splade-cocondenser-ensembledistil",
    # ),
}

ALL_MODELS = list(MODEL_MAP.values())

def predict(args):
    dataset = load_dataset("orionweller/NevIR", args.split)[args.split]
    df = dataset.to_pandas()

    if args.models == ["all"]:
        selected_models = ALL_MODELS        
    else:
        selected_models = []
        for model in args.models:
            selected_models.append(MODEL_MAP[model])

    model_bar = tqdm.tqdm(selected_models, leave=True)
    for (model_name, model_func, model_details) in model_bar:
        model_bar.set_description(f"Model {model_name}")
        model_results = []
        row_p_bar = tqdm.tqdm(df.iterrows())
        model = None
        for (row_idx, row) in row_p_bar:  # each instance
            if model_details is not None:
                results, model, sim_score = model_func(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    model_details,
                    model=model,
                )
                model_results.append(results)
            else:
                results, model, sim_score = model_func(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    model=model,
                )
                model_results.append(results)

        model_df = pd.DataFrame(model_results)
        model_df["q1_probs"] = model_df.q1.apply(lambda x: softmax(x))
        model_df["q2_probs"] = model_df.q2.apply(lambda x: softmax(x))
        model_df["pairwise_score"] = model_df.score.apply(lambda x: x == 1.0)
        overall_score = model_df.pairwise_score.mean()
        print(
            f'For model {model_name} the average score is {overall_score}'
        )

        if args.output is not None:
            model_df.to_csv(os.path.join(args.output, f"results.csv"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        help="which split to evaluate on",
        type=str,
        default="validation",
    )
    parser.add_argument(
        "-m",
        "--models",
        help="name of the model to predict with or all models (default). Options include tfidf, colbertv1, dpr, or other dense sentencetransformers models",
        nargs="*",
        default=["all"],
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output directory to save results",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    predict(args)
