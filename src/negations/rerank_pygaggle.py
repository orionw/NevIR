from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
import random
from transformers import set_seed
import numpy as np
import pandas as pd
import argparse
import torch
from transformers import T5ForConditionalGeneration

set_seed(1)


def calc_preferred_rerankers(
    doc1, doc2, q1, q2, model_name="castorini/monot5-base-msmarco-10k", model=None
):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        query1, query2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model_name: string containing the type of model to run
        model: the preloaded model, if caching

    Returns:
        A dictionary containing each query (q1 or q2) and the score (P@1) for the pair

    """
    if model is None:
        ### Model initialization
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        reranker = MonoT5(model=model)
    else:
        reranker = model
    # Option 2: here's what Pyserini would have retrieved, hard-coded
    passages = [[1, doc1], [2, doc2]]
    texts = [
        Text(p[1], {"docid": p[0]}, 0) for p in passages
    ]  # Note, pyserini scores don't matter since T5 will ignore them.

    queries = [q1, q2]
    results = {}
    num_correct = 0
    # reranked = reranker.rerank(Query(doc1), [Text(doc2, {"docid": 2}, 0)])
    # breakpoint()
    similarity_score = None
    
    for idx, query in enumerate(queries):
        reranked = reranker.rerank(Query(query), texts)
        first_score_index = [
            idx
            for idx, item in enumerate(reranked)
            if item.metadata["docid"] == 1
        ][0]
        second_score_index = [
            idx
            for idx, item in enumerate(reranked)
            if item.metadata["docid"] == 2
        ][0]
        scores = [
            reranked[first_score_index].score,
            reranked[second_score_index].score,
        ]
        results[f"q{idx+1}"] = scores
        should_be_higher = scores[idx]
        should_be_lower = scores[0] if idx != 0 else scores[1]

        if type(should_be_higher) == np.ndarray and len(should_be_higher) > 1:
            should_be_higher = should_be_higher[1]  # entailment models
            should_be_lower = should_be_lower[1]

        if should_be_higher > should_be_lower:
            num_correct += 1
    model = reranker

    results["score"] = num_correct / 2
    return results, model, similarity_score


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
        "-m",
        "--model_name",
        help="the model to use, if not loading from file",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    if not args.file and (
        not args.doc1 or not args.doc2 or not args.q1 or not args.q2
    ):
        print(
            "Error: need either a file path or the input args (d1, d2, q1, q2, model)"
        )
    elif args.file:
        print("Loading from file...")
        df = pd.read_csv(args.file)
        # breakpoint()
        for (idx, row) in df.iterrows():
            print(
                calc_preferred_rerankers(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    row["model_name"],
                )[0]
            )
    else:
        print("Loading from args...")
        print(
            calc_preferred_rerankers(
                args.doc1, args.doc2, args.q1, args.q2, args.model_name
            )[0]
        )
