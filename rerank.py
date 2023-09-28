from sentence_transformers import (
    SentenceTransformer,
    util,
    CrossEncoder # can use later if we want a real cross-encoder for reranking
)
import torch

torch.manual_seed(1)

# corpus would be what was returned from Lucene
corpus = [
    'The girl is carrying a baby.',
    'A man is riding a horse.',
    'A woman is playing violin.',
    'Two men pushed carts through the woods.',
    'A man is riding a white horse on an enclosed ground.',
    'A monkey is playing drums.',
    'A cheetah is running behind its prey.',
    'A man is eating food.',
    'A man is eating a piece of bread.',
]

query = "A man is eating pasta" # this would be query, but here for example

### NOTE: these lines will be slowish without a GPU but we can see how they go
embedder = SentenceTransformer("orionweller/test") # I didn't want to be too specific on huggingface, will update later
passage_embeddings = embedder.encode(corpus)
query_embedding = embedder.encode(query)
scores = util.dot_score(query_embedding, passage_embeddings)[0] # only one query

# use these scores to rerank the corpus
top_results = torch.topk(scores, k=5).indices # or k=whatever we want to show the user
reranked_corpus = [corpus[i] for i in top_results] # this could be done vectorized I think, probably is slow
print(reranked_corpus) # output below
"""
[
    'A man is eating a piece of bread.',
    'A man is eating food.',
    'A man is riding a horse.',
    'A woman is playing violin.',
    'A monkey is playing drums.'
]
"""