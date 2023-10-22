import math
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

sns.set_style("darkgrid")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")

seed = 97
random.seed(seed)
np.random.seed(seed)

# Load the dataset
df = pd.read_excel("/Users/berwyn/Documents/XJTLU/Semester_2/IOM414 Big Data-Applications in Business/Coursework/data/Online Retail.xlsx")
df.head()

# Handling Missing Values
df.isnull().sum()

# Drop rows with null CustomerID
df.dropna(subset=["CustomerID"], inplace=True)

# Fill missing values in Description column with empty string
df["Description"].fillna("", inplace=True)

df.isnull().sum().sum()

# Treating Canceled Invoices
size_before = len(df)

# Identify negative quantities
neg_quantity = df[df["Quantity"] < 0][["CustomerID", "StockCode", "Quantity"]].sort_values("Quantity")
print(f"Negative Quantity: {len(neg_quantity)}")

filtered = df[df["CustomerID"].isin(neg_quantity["CustomerID"])]
filtered = filtered[filtered["StockCode"].isin(neg_quantity["StockCode"])]

pos_counters = []
for idx, series in neg_quantity.iterrows():
    customer = series["CustomerID"]
    code = series["StockCode"]
    quantity = -1 * series["Quantity"]
    counterpart = filtered[(filtered["CustomerID"] == customer) & (filtered["StockCode"] == code) & (filtered["Quantity"] == quantity)]
    pos_counters.extend(counterpart.index.to_list())

to_drop = neg_quantity.index.to_list() + pos_counters
df.drop(to_drop, axis=0, inplace=True)
print(f"Removed {size_before - len(df)} rows from the dataset")

(df["Quantity"] <= 0).sum()

df.drop(df[df["UnitPrice"] == 0].index, axis=0, inplace=True)

# Feature Engineering and Preprocessing
# Extracting time-related features from InvoiceDate
df["InvoiceDateDay"] = df["InvoiceDate"].dt.date
df["InvoiceDateTime"] = df["InvoiceDate"].dt.time
df["InvoiceYear"] = df["InvoiceDate"].dt.year
df["InvoiceMonth"] = df["InvoiceDate"].dt.month
df["InvoiceMonthName"] = df["InvoiceDate"].dt.month_name()
df["InvoiceDay"] = df["InvoiceDate"].dt.day
df["InvoiceDayName"] = df["InvoiceDate"].dt.day_name()
df["InvoiceDayOfWeek"] = df["InvoiceDate"].dt.day_of_week
df["InvoiceWeekOfYear"] = df["InvoiceDate"].dt.weekofyear
df["InvoiceHour"] = df["InvoiceDate"].dt.hour
df["TotalValue"] = df["Quantity"] * df["UnitPrice"]

# RFM Analysis
ref_date = df["InvoiceDateDay"].max() + timedelta(days=1)
df_customers = df.groupby("CustomerID").agg({
    "InvoiceDateDay": lambda x: (ref_date - x.max()).days,
    "InvoiceNo": "count",
    "TotalValue": "sum"
}).rename(columns={
    "InvoiceDateDay": "Recency",
    "InvoiceNo": "Frequency",
    "TotalValue": "MonetaryValue"
})
df_customers.head(10)
df_customers.info()

# Removing Outliers
def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered


df_customers = remove_outliers(df_customers, "Recency")
df_customers = remove_outliers(df_customers, "Frequency")
df_customers = remove_outliers(df_customers, "MonetaryValue")

df_customers.info()

# Exploratory Data Analysis
# Revenue by Country
plt.figure(figsize=(12, 6))
df.groupby("Country")["TotalValue"].sum().sort_values(ascending=False).plot(kind="bar")
plt.ylabel("Revenue")
plt.title("Revenue by Country")
plt.show()

# Revenue by Month
plt.figure(figsize=(12, 6))
df.groupby("InvoiceMonthName")["TotalValue"].sum().plot(kind="bar")
plt.ylabel("Revenue")
plt.title("Revenue by Month")
plt.show()

# Revenue by Day of Week
plt.figure(figsize=(12, 6))
df.groupby("InvoiceDayName")["TotalValue"].sum().plot(kind="bar")
plt.ylabel("Revenue")
plt.title("Revenue by Day of Week")
plt.show()

# Top 10 Customers by Revenue
plt.figure(figsize=(12, 6))
df.groupby("CustomerID")["TotalValue"].sum().sort_values(ascending=False)[:10].plot(kind="bar")
plt.ylabel("Revenue")
plt.title("Top 10 Customers by Revenue")
plt.show()

# Customer Segmentation using K-means Clustering
X = df_customers[["Recency", "Frequency", "MonetaryValue"]]

# Feature Scaling and Transformation
transformer = ColumnTransformer(
    transformers=[
        ("power_transformer", PowerTransformer(method="yeo-johnson", standardize=True),
         ["Recency", "Frequency", "MonetaryValue"])
    ]
)

scaler = StandardScaler()

# Create K-means pipeline
pipeline = Pipeline([
    ("preprocessing", transformer),
    ("scaling", scaler),
    ("kmeans", KMeans(n_clusters=2, random_state=seed))
])

# Determine the optimal number of clusters using Elbow Method
k_values = range(1, 11)
inertias = []

for k in k_values:
    pipeline.set_params(kmeans__n_clusters=k)
    pipeline.fit(X)
    inertias.append(pipeline.named_steps["kmeans"].inertia_)

plt.figure(figsize=(12, 6))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Determining the Optimal Number of Clusters")
plt.show()

# Silhouette Analysis
silhouette_scores = []

for k in k_values:
    pipeline.set_params(kmeans__n_clusters=k)
    pipeline.fit(X)
    labels = pipeline.named_steps["kmeans"].labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(12, 6))
plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis - Determining the Optimal Number of Clusters")
plt.show()

# Running K-means with K=2
pipeline.set_params(kmeans__n_clusters=2)
pipeline.fit(X)

# Assign cluster labels to customers
cluster_labels = pipeline.predict(X)
df_customers["Cluster"] = cluster_labels

# Visualize the clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_customers, x="Frequency", y="MonetaryValue", hue="Cluster", palette="Set1")
plt.xlabel("Frequency")
plt.ylabel("Monetary Value")
plt.title("Customer Segmentation - K-means Clustering")
plt.show()

# Analyze cluster characteristics
cluster_summary = df_customers.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "MonetaryValue": "mean"
})
cluster_summary

# Get cluster assignments for all customers in the original dataset
df["Cluster"] = cluster_labels[df["CustomerID"].astype(int)]

# Update: Additional Analysis
# Visualize cluster distribution by country
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="Country", hue="Cluster", palette="Set1")
plt.xlabel("Country")
plt.ylabel("Count")
plt.title("Cluster Distribution by Country")
plt.xticks(rotation=90)
plt.show()

# Visualize cluster distribution by month
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="InvoiceMonthName", hue="Cluster", palette="Set1", order=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
plt.xlabel("Month")
plt.ylabel("Count")
plt.title("Cluster Distribution by Month")
plt.xticks(rotation=90)
plt.show()

# Visualize cluster distribution by day of week
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="InvoiceDayName", hue="Cluster", palette="Set1", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.xlabel("Day of Week")
plt.ylabel("Count")
plt.title("Cluster Distribution by Day of Week")
plt.xticks(rotation=90)
plt.show()

# Visualize cluster distribution by hour
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="InvoiceHour", hue="Cluster", palette="Set1")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.title("Cluster Distribution by Hour")
plt.show()

# Update: Customer Profiling
# Profiling the top customers in each cluster
top_customers = df_customers.groupby("Cluster").apply(lambda x: x.nlargest(3, "MonetaryValue"))
top_customers

# Update: Customer Segmentation Analysis
# Average Monetary Value by Cluster
plt.figure(figsize=(12, 6))
sns.barplot(data=df_customers, x="Cluster", y="MonetaryValue", ci=None)
plt.xlabel("Cluster")
plt.ylabel("Average Monetary Value")
plt.title("Average Monetary Value by Cluster")
plt.show()

# Average Frequency by Cluster
plt.figure(figsize=(12, 6))
sns.barplot(data=df_customers, x="Cluster", y="Frequency", ci=None)
plt.xlabel("Cluster")
plt.ylabel("Average Frequency")
plt.title("Average Frequency by Cluster")
plt.show()

# Average Recency by Cluster
plt.figure(figsize=(12, 6))
sns.barplot(data=df_customers, x="Cluster", y="Recency", ci=None)
plt.xlabel("Cluster")
plt.ylabel("Average Recency")
plt.title("Average Recency by Cluster")
plt.show()


