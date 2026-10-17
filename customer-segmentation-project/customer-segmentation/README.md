# Customer Segmentation Using K-Means

A university-level Data Science portfolio project using **Python, Pandas, NumPy, Matplotlib and Scikit-learn**. It groups customers by purchasing behavior (RFM-style features) using K-Means, compares cluster counts using the elbow method and silhouette score, and visualizes clusters with PCA.

## Dataset
`data/customers.csv` contains **300 synthetic customers**, generated for learning. **These are not real customers or verified business insights.** Fields: `CustomerID`, `RecencyDays` (days since last order), `Frequency` (number of orders), `AvgOrderValue`, `Monetary` (total spend). The monetary values have no real currency.

## Run on Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.analysis
.\.venv\Scripts\python.exe -m jupyter notebook
```
Open `notebooks/customer_segmentation.ipynb` and run cells from top to bottom.

## Methods
1. Inspect missing values and feature distributions.
2. Select RecencyDays, Frequency, Monetary; standardize to avoid dominance by larger scales.
3. Compare K=2 to K=8 using elbow (inertia) and silhouette.
4. Choose K with the highest silhouette score **for this educational example**, fit K-Means with a fixed random seed.
5. Visualize a 2D PCA projection (PCA is for visualization, not the clustering input).
6. Interpret segment averages. Cluster numbers are arbitrary and do not imply quality ranking.

## Outputs
`reports/figures/elbow.png`, `silhouette.png`, `clusters_pca.png`, `reports/customer_segments.csv`, `reports/cluster_summary.csv`. Optional SQL queries are in `sql/queries.sql`.

## Limitations
The data are artificially constructed with customer groups, so clusters can appear clearer than real-world customer segments. Silhouette does not prove business value. For a stronger portfolio, repeat on a public retail dataset with customer-level transactions and document data cleaning, RFM calculation, and privacy considerations.

## CV (after completing and understanding the work)
**Customer Segmentation Using K-Means** — Analyzed customer purchasing behavior using Pandas; standardized RFM features; compared K-Means cluster counts with elbow and silhouette methods; visualized clusters using PCA and interpreted segment-level statistics.
