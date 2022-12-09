# CS_paper
This code contains a duplicate detection method for identifying duplicate products across different web shops.
After data cleaning and extracting information from the product titles, the products are represented by binary vectors of their attributes.
LSH is applied to minhash signatures of the binary vectors to identify candidate pairs.
To assess whether these pairs are duplicates, hierarcical agglomerative clustering is performed on the dissimilarity matrix. This matrix contains Jaccard similarities between candidate pairs.
