import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import argparse


def calc_preferred_tfidf(doc1, doc2, q1, q2, model=None):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        query1, query2: strings for queries that are only relevant to the corresponding doc (doc1 -> q1, doc2 -> q2)
        model: does nothing for tfidf

    Returns:
        A dictionary containing each query (q1 or q2) and the scores for the documents

    """
    results = {}
    num_correct = 0
    stemmer = nltk.stem.porter.PorterStemmer()
    remove_punctuation_map = dict(
        (ord(char), None) for char in string.punctuation
    )

    def stem_tokens(tokens):
        return [stemmer.stem(item) for item in tokens]

    """remove punctuation, lowercase, stem"""

    def normalize(text):
        return nltk.word_tokenize(
            text.lower().translate(remove_punctuation_map)
        )

    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words="english")

    def cosine_sim(text1, text2):
        tfidf = vectorizer.fit_transform([text1, text2])
        return ((tfidf * tfidf.T).A)[0, 1]

    for idx, query in enumerate([q1, q2]):
        q_to_d1 = cosine_sim(query, doc1)
        q_to_d2 = cosine_sim(query, doc2)
        scores = [q_to_d1, q_to_d2]
        results[f"q{idx+1}"] = scores
        should_be_higher = scores[idx]
        should_be_lower = scores[0] if idx != 0 else scores[1]
        if should_be_higher > should_be_lower:
            num_correct += 1

    results["score"] = num_correct / 2
    return results, vectorizer.vocabulary, cosine_sim(doc1, doc2)


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
                calc_preferred_tfidf(
                    row["doc1"], row["doc2"], row["q1"], row["q2"]
                )[0]
            )
    else:
        print(calc_preferred_tfidf(args.doc1, args.doc2, args.q1, args.q2)[0])
