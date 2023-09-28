import nltk
import string
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
import pandas as pd
import argparse

import numpy as np


def calc_preferred_splade(
    doc1,
    doc2,
    q1,
    q2,
    specific_model: str = "naver/splade-cocondenser-ensembledistil",
    model=None,
):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        query1, query2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model: splade model

    Returns:
        A dictionary containing each query (q1 or q2) and the scores for the documents

    """
    if model is None:
        model = Splade(specific_model, agg="max")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(specific_model)
    else:
        model, tokenizer = model

    results = {}
    num_correct = 0

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        doc1_tok = tokenizer(doc1, return_tensors="pt")
        doc2_tok = tokenizer(doc2, return_tensors="pt")
        q1_tok = tokenizer(q1, return_tensors="pt")
        q2_tok = tokenizer(q2, return_tensors="pt")

        if torch.cuda.is_available():
            doc1_tok = {k: v.cuda() for k, v in doc1_tok.items()}
            doc2_tok = {k: v.cuda() for k, v in doc2_tok.items()}
            q1_tok = {k: v.cuda() for k, v in q1_tok.items()}
            q2_tok = {k: v.cuda() for k, v in q2_tok.items()}

        doc1_rep = model(d_kwargs=doc1_tok)[
            "d_rep"
        ].squeeze()
        doc2_rep = model(d_kwargs=doc2_tok)[
            "d_rep"
        ].squeeze()
        q1_rep = model(d_kwargs=q1_tok)[
            "d_rep"
        ].squeeze()
        q2_rep = model(d_kwargs=q2_tok)[
            "d_rep"
        ].squeeze()

    for idx, query_rep in enumerate([q1_rep, q2_rep]):
        q_to_d1 = torch.dot(doc1_rep, query_rep)  # pylint: disable=no-member
        q_to_d2 = torch.dot(doc2_rep, query_rep)  # pylint: disable=no-member
        scores = [q_to_d1.item(), q_to_d2.item()]
        results[f"q{idx+1}"] = scores
        should_be_higher = scores[idx]
        should_be_lower = scores[0] if idx != 0 else scores[1]
        if should_be_higher > should_be_lower:
            num_correct += 1

    results["score"] = num_correct / 2
    return results, (model, tokenizer), torch.nn.functional.cosine_similarity(doc1_rep.unsqueeze(0), doc2_rep.unsqueeze(0)).item()


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
                calc_preferred_splade(
                    row["doc1"], row["doc2"], row["q1"], row["q2"]
                )[0]
            )
    else:
        print("Loading from args...")
        print(calc_preferred_splade(args.doc1, args.doc2, args.q1, args.q2)[0])
