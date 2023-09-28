from sentence_transformers import (
    SentenceTransformer,
    util,
    CrossEncoder,
    models,
)
import torch
from transformers import set_seed
import numpy as np
import pandas as pd
import argparse
import tqdm
import json

set_seed(1)


def calc_preferred_dense(doc1, doc2, q1, q2, model_name="dpr", model=None):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        query1, query2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model_name: string containing the type of model to run
        model: the preloaded model, if caching
        return_similarity: whether to return the similarity score or not

    Returns:
        A dictionary containing each query (q1 or q2) and the score (P@1) for the pair

    """

    ### Model initialization
    if model_name == "dpr":
        model_type = "dpr"
        if model is not None:
            passage_encoder, query_encoder = model
        else:
            passage_encoder = SentenceTransformer(
                "facebook-dpr-ctx_encoder-multiset-base"
            )
            query_encoder = SentenceTransformer(
                "facebook-dpr-question_encoder-multiset-base"
            )
    elif "cross-encoder" in model_name or "t5" in model_name:
        model_type = "cross_encoder"
        if "t5" in model_name:
            word_embedding_model = models.Transformer(
                "t5-base", max_seq_length=256
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension()
            )
            model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model]
            )

        if model is None:
            model = CrossEncoder(model_name)

    else:
        model_type = "biencoder"
        if model is not None:
            embedder = model
        else:
            embedder = SentenceTransformer(model_name)            

    corpus = [doc1, doc2]
    queries = [q1, q2]
    results = {}
    num_correct = 0
    doc_sim = None

    ### Do Retrieval
    if model_type == "dpr":
        passage_embeddings = passage_encoder.encode(corpus)

        query_encoder = SentenceTransformer(
            "facebook-dpr-question_encoder-single-nq-base"
        )
        doc_sim = torch.nn.functional.cosine_similarity(torch.tensor(passage_embeddings[0]).unsqueeze(0), torch.tensor(passage_embeddings[1]).unsqueeze(0))

        for idx, query in enumerate(queries):
            query_embedding = query_encoder.encode(query)
            # must use dot-product, not cosine_similarity
            scores = util.dot_score(query_embedding, passage_embeddings)[
                0
            ]  # only one query
            results[f"q{idx+1}"] = scores.tolist()
            should_be_higher = scores[idx]
            should_be_lower = scores[0] if idx != 0 else scores[1]
            if should_be_higher > should_be_lower:
                num_correct += 1
        model = (passage_encoder, query_encoder)

    elif model_type == "cross_encoder":
        doc_sim = None # model.predict([[doc1, doc2]]).tolist()[0]
        for idx, query in enumerate(queries):
            scores = model.predict([[query, doc1], [query, doc2]])
            if len(scores[0]) > 2: # NLI models
                scores = [
                    scores[0][1], # entailment
                    scores[1][1] # entailment
                ]
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

    else:  # bi-encoder that is not DPR
        corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
        doc_sim = torch.nn.functional.cosine_similarity(corpus_embeddings[0].unsqueeze(0), corpus_embeddings[1].unsqueeze(0))

        for idx, query in enumerate(queries):
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            scores = util.dot_score(query_embedding, corpus_embeddings)[0].cpu()
            results[f"q{idx+1}"] = scores.tolist()
            should_be_higher = scores[idx]
            should_be_lower = scores[0] if idx != 0 else scores[1]
            if should_be_higher > should_be_lower:
                num_correct += 1
        model = embedder

    results["score"] = num_correct / 2
    return results, model, doc_sim


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
        scores = []
        sim_scores = []
        model = None
        global_model_name = args.model_name if args.model_name is not None else None
        for (idx, row) in tqdm.tqdm(df.iterrows(), total=len(df)):
            model_name = global_model_name if global_model_name is not None else row["model_name"]
            results, model, sim_score = calc_preferred_dense(
                    row["doc1"],
                    row["doc2"],
                    row["q1"],
                    row["q2"],
                    model_name,
                    model=model
                )
            scores.append(
                results["score"]
            )
            sim_scores.append(sim_score.item())
        print((np.array(scores) == 1).mean())
        if global_model_name is not None:
            model_name = global_model_name.split("/")[-3] + "-" + global_model_name.split("/")[-1]
            
        with open(args.file.replace("csv", f"{model_name.replace('/', '_')}.results"), "w") as f:
            f.write(json.dumps({
                "scores": scores,
                "sim_scores": np.mean(sim_scores),
                "sim_scores_std": np.std(sim_scores),
                "paired_accuracy": (np.array(scores) == 1).mean(),
                "model": model_name
            }))
        print(args.file.replace("csv", f"{model_name.replace('/', '_')}.results"))
    else:
        print("Loading from args...")
        print(
            calc_preferred_dense(
                args.doc1, args.doc2, args.q1, args.q2, args.model_name
            )[0]
        )
