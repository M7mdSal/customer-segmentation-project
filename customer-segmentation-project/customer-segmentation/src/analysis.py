"""Educational customer segmentation: K-Means, elbow method and PCA."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customers.csv"
OUT = ROOT / "reports" / "figures"
FEATURES = ["RecencyDays", "Frequency", "Monetary"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    print("Rows:", len(df), "Missing values:", df.isna().sum().sum())
    if df[FEATURES].isna().any().any():
        raise ValueError("Clean missing values before clustering")
    x = StandardScaler().fit_transform(df[FEATURES])
    ks = range(2, 9)
    inertia, scores = [], []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x)
        inertia.append(model.inertia_)
        scores.append(silhouette_score(x, labels))
    fig, ax = plt.subplots()
    ax.plot(list(ks), inertia, marker="o")
    ax.set(xlabel="Number of clusters (k)", ylabel="Inertia", title="Elbow Method")
    fig.tight_layout(); fig.savefig(OUT / "elbow.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(list(ks), scores, marker="o")
    ax.set(xlabel="Number of clusters (k)", ylabel="Silhouette score", title="Silhouette Comparison")
    fig.tight_layout(); fig.savefig(OUT / "silhouette.png", dpi=150); plt.close(fig)
    best_k = list(ks)[scores.index(max(scores))]
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(x)
    coords = PCA(n_components=2).fit_transform(x)
    fig, ax = plt.subplots()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=df["Cluster"], cmap="viridis", s=24)
    ax.set(xlabel="Principal Component 1", ylabel="Principal Component 2", title="Customer Clusters (PCA projection)")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout(); fig.savefig(OUT / "clusters_pca.png", dpi=150); plt.close(fig)
    summary = df.groupby("Cluster")[FEATURES].agg(["mean", "count"])
    df.to_csv(ROOT / "reports" / "customer_segments.csv", index=False)
    summary.to_csv(ROOT / "reports" / "cluster_summary.csv")
    print("Selected k:", best_k, "Silhouette:", round(max(scores), 3))
    print(df.groupby("Cluster")[FEATURES].mean().round(2))
    print("Saved results in reports/")

if __name__ == "__main__":
    main()
