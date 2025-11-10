import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import argparse

"""
    Here we're going to create automatic themed clusters using TF-IDF + KMeans.

"""

def main(csv_path, k):
    df = pd.read_csv(csv_path)

    # drop empty comments or very short ones
    df["commentaire"] = df["commentaire"].fillna("").astype(str).str.strip()
    df = df[df["commentaire"].str.len() > 5].copy()
    texts = df["commentaire"].astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        min_df=3,
        max_df=0.8,
        sublinear_tf=True #to lower the weight of words that show too often (before this, we had clusters with 15k verbatims and others with 700)
    )

    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"Silhouette score: {sil:.3f}")

    df["cluster"] = labels

    terms = np.array(vectorizer.get_feature_names_out())
    for c in range(k):
        idx = np.where(labels==c)[0]
        center = kmeans.cluster_centers_[c]
        top_terms = terms[center.argsort()[-10:][::-1]]
        print(f"\n Cluster {c} ({len(idx)} commentaires)")
        print("Top terms:", ", ".join(top_terms))

    df.to_csv("data/csat_with_clusters.csv", index=False)
    print(" Clusters saved in data/csat_with_clusters.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/csat.csv")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    main(args.csv, args.k)
