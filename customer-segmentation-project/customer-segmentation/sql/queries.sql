-- Optional MySQL exercise: import data/customers.csv into a customers table first.
SELECT COUNT(*) AS total_customers FROM customers;
SELECT AVG(Monetary) AS avg_monetary, AVG(Frequency) AS avg_frequency FROM customers;
SELECT CustomerID, Monetary FROM customers ORDER BY Monetary DESC LIMIT 10;
-- After exporting reports/customer_segments.csv as a table named customer_segments:
SELECT Cluster, COUNT(*) AS customers, ROUND(AVG(Monetary),2) AS avg_spending
FROM customer_segments GROUP BY Cluster ORDER BY avg_spending DESC;
