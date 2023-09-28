import os
import sys
import pandas as pd
import argparse

sys.path.insert(1, "ColBERT/.")  # from git clone of their repo
from argparse import Namespace
import __main__
import random
from collections import OrderedDict
import random
import torch

# from colbert.utils.utils import print_message  # pylint: disable=import-error
from colbert.modeling.inference import (  # pylint: disable=import-error
    ModelInference,
)
from colbert.evaluation.slow import slow_rerank  # pylint: disable=import-error
from colbert.evaluation.loaders import (  # pylint: disable=import-error
    load_colbert,
)

from transformers import set_seed

set_seed(42)


def my_evaluate(colbert, queries, passages, make_heatmap=False):
    all_results = {}
    num_correct = 0

    with torch.no_grad():
        inference = ModelInference(colbert, amp=True)

        for query_idx, query in enumerate(queries):
            Q = inference.queryFromText([query])

            D_ = inference.docFromText(passages, bsize=2)

            scores = colbert.score(Q, D_).cpu().tolist()

            all_results[f"q{query_idx+1}"] = scores
            should_be_higher = scores[query_idx]
            should_be_lower = scores[0] if query_idx != 0 else scores[1]
            # print(f"q{query_idx+1}: {should_be_higher} > {should_be_lower} for {scores}")
            if should_be_higher > should_be_lower:
                num_correct += 1

    all_results["score"] = num_correct / 2
    # print(all_results)
    return all_results


def calc_preferred_colbert(
    doc1, doc2, q1, q2, model_version: str = "v1", model=None, make_heatmap=False
):
    """
    Re-ranks ColBERT V1 to avoid caching and indexing, adapted from the ColBERT repo, file `test.py`

    Input:
        doc1, doc2: strings containing the documents/passages
        query1, query2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model_name: string containing the type of model to run
        model: the preloaded model, if caching
        make_heatmap: whether to make a heatmap of the MaxSim matrix

    Returns:
        A dictionary containing each query (q1 or q2) and the score (P@1) for the pair

    """
    # default from parser.py needed to load the model
    path_to_weights = (
        "/exp/oweller/NegationInIR/colbert_weights/colbert-v1.dnn"
        if model_version == "v1"
        else "/exp/oweller/NegationInIR/colbert_weights/colbertv2.0/pytorch_model.bin"
    )
    if "/" in model_version:
        path_to_weights = model_version

    fake_model_args = {
        "query_maxlen": 32,
        "doc_maxlen": 180,
        "dim": 128,
        "similarity": "l2",
        "amp": True,
        "rank": 0,
        "checkpoint": path_to_weights,
        "mask_punctuation": True,
    }
    if model is not None:
        colbert = model
    else:
        colbert, _ = load_colbert(Namespace(**fake_model_args))

    return my_evaluate(colbert, [q1, q2], [doc1, doc2], make_heatmap), colbert, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        help="whether to load a file and if so, the path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d1",
        "--doc1",
        help="doc1 if loading from command line",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d2",
        "--doc2",
        help="doc2 if loading from command line",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-q1",
        "--q1",
        help="q1 if loading from command line",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-q2",
        "--q2",
        help="q1 if loading from command line",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--version",
        help="whether to load v1 or v2",
        type=str,
        default="v1",
    )
    args = parser.parse_args()
    if not args.file and (
        not args.doc1 or not args.doc2 or not args.q1 or not args.q2
    ):
        print(
            "Error: need either a file path or the input args (d1, d2, q1, q2)"
        )
    elif args.file:
        print("Loading from file...")
        df = pd.read_csv(args.file)
        for (idx, row) in df.iterrows():
            print(
                calc_preferred_colbert(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    row["version"],
                )[0]
            )
    else:
        print("Loading from args...")
        print(
            calc_preferred_colbert(
                args.doc1, args.doc2, args.q1, args.q2, args.version
            )[0]
        )
