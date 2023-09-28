mkdir -p downloads/

# ColBERTv2 checkpoint trained on MS MARCO Passage Ranking (388MB compressed)
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -P downloads/
tar -xvzf downloads/colbertv2.0.tar.gz -C downloads/
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv1/colbertv1.0.tar.gz -P downloads/
