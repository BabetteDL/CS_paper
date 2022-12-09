# CS_paper
This code contains a duplicate detection method for identifying similar products across different web shops.
After data cleaning, the products are represented by binary vectors of their attributes.
LSH is applied to minhash signatures of the binary vectors to identify candidate pairs.
To assess whether these pairs are duplicates, hierarcical clustering is performed based on the Jaccard similarity between two products.
