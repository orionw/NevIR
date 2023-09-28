from transformers import set_seed
import numpy as np
import pandas as pd
import argparse
import rocketqa
import torch

set_seed(1)


def calc_preferred_rocketqa(
    doc1, doc2, q1, q2, model_name="v1_marco_de", model=None
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
    ### Model initialization
    if "_ce" in model_name:
        # model_type = "cross_encoder"
        if model is not None:
            model = model
        else:
            model = rocketqa.load_model(
                model=model_name, use_cuda=False, batch_size=16
            )

    elif "_de" in model_name:
        # model_type = "biencoder"
        if model is not None:
            embedder = model
        else:
            embedder = rocketqa.load_model(
                model=model_name, use_cuda=False, batch_size=16
            )

    corpus = [doc1, doc2]
    queries = [q1, q2]
    results = {}
    num_correct = 0

    ### Do Retrieval
    if "_de" in model_name:
        corpus_rep = list(embedder.encode_para(corpus))
        similarity_score = torch.nn.functional.cosine_similarity(torch.tensor(corpus_rep[0]).unsqueeze(0), torch.tensor(corpus_rep[1]).unsqueeze(0))
        for idx, query in enumerate(queries):
            scores = list(embedder.matching(query=[query] * 2, para=corpus))
            results[f"q{idx+1}"] = scores
            should_be_higher = scores[idx]
            should_be_lower = scores[0] if idx != 0 else scores[1]
            if should_be_higher > should_be_lower:
                num_correct += 1
        model = embedder

    elif "_ce" in model_name:
        similarity_score = None
        for idx, query in enumerate(queries):
            scores = list(
                model.matching(
                    query=[query, query], para=corpus, title=["", ""]
                )
            )
            results[f"q{idx+1}"] = scores
            should_be_higher = scores[idx]
            should_be_lower = scores[0] if idx != 0 else scores[1]

            if (
                type(should_be_higher) == np.ndarray
                and len(should_be_higher) > 1
            ):
                should_be_higher = should_be_higher[1]  # entailment models
                should_be_lower = should_be_lower[1]

            if should_be_higher > should_be_lower:
                num_correct += 1
        model = model
    else:
        raise Exception()

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
        default="dpr",
    )
    args = parser.parse_args()
    if not args.file and (
        not args.doc1
        or not args.doc2
        or not args.q1
        or not args.q2
        or args.model
    ):
        print(
            "Error: need either a file path or the input args (d1, d2, q1, q2, model)"
        )
    elif args.file:
        print("Loading from file...")
        df = pd.read_csv(args.file)
        for (idx, row) in df.iterrows():
            print(
                calc_preferred_rocketqa(
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
            calc_preferred_rocketqa(
                args.doc1, args.doc2, args.q1, args.q2, args.model_name
            )[0]
        )
