import math
import random
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta


sns.set_style("darkgrid")


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")

seed = 97
random.seed(seed)
np.random.seed(seed)

df = pd.read_excel("/Users/berwyn/Documents/XJTLU/Semester_2/IOM414 Big Data-Applications in Business/Coursework/data/Online Retail.xlsx")
df.head()

# Handling Missing Values
df.isnull().sum()
df.drop(df[df["CustomerID"].isnull()].index, axis=0, inplace=True)
df["Description"] = df["Description"].fillna("")
df.isnull().sum().sum()

# Treating Canceled Invoices
size_before = len(df)
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
# extracting time related features from InvoiceDate
df["InvoiceDateDay"] = df["InvoiceDate"].dt.date
df["InvoiceDateTime"] = df["InvoiceDate"].dt.time
df["InvoiceYear"] = df["InvoiceDate"].dt.year
df["InvoiceMonth"] = df["InvoiceDate"].dt.month
df["InvoiceMonthName"] = df["InvoiceDate"].dt.month_name()
df["InvoiceDay"] = df["InvoiceDate"].dt.day
df["InvoiceDayName"] = df["InvoiceDate"].dt.day_name()
df["InvoiceDayOfWeek"] = df["InvoiceDate"].dt.day_of_week
# df["InvoiceWeekOfYear"] = df["InvoiceDate"].dt.weekofyear()
df["InvoiceHour"] = df["InvoiceDate"].dt.hour
df["TotalValue"] = df["Quantity"] * df["UnitPrice"]

# RFM Analysis
ref_date = df["InvoiceDateDay"].max() + timedelta(days=1)

df_customers = df.groupby("CustomerID").agg({
    "InvoiceDateDay": lambda x : (ref_date - x.max()).days,
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
n_cols = len(df_customers.columns)
fig, axes = plt.subplots(n_cols, 2, figsize=(16, n_cols * 4))

for i, col in enumerate(df_customers.columns):
    sns.boxplot(data=df_customers, y=col, ax=axes[i][0])
    sns.histplot(data=df_customers, x=col, kde=True, ax=axes[i][1])

fig.show()


def remove_outliers(df, col, threshold=1.5):
    Q1 = np.quantile(df[col], .25)
    Q3 = np.quantile(df[col], .75)
    IQR = Q3 - Q1
    df.drop(df[(df[col] < (Q1 - threshold * IQR)) | (df[col] > (Q3 + threshold * IQR))].index, axis=0, inplace=True)

    return df


for col in df_customers.columns:
    size_before = len(df_customers)
    df_customers = remove_outliers(df_customers, col)
    print(f"Removed {size_before - len(df_customers)} outliers from {col}")

n_cols = len(df_customers.columns)
fig, axes = plt.subplots(n_cols, 2, figsize=(16, n_cols * 4))

for i, col in enumerate(df_customers.columns):
    sns.boxplot(data=df_customers, y=col, ax=axes[i][0])
    sns.histplot(data=df_customers, x=col, kde=True, ax=axes[i][1])

fig.show()

# Exploratory Data Analysis
# Revenue by Country
country_revenue = df.groupby("Country")["TotalValue"].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(12, 4))
plt.title("Total Revenue by Country")
sns.barplot(data=country_revenue, x="Country", y="TotalValue")
plt.ylabel("Total Revenue")
plt.xticks(rotation=90)
plt.show()

# Revenue by Month of the Year
revenue_month = df.groupby(["InvoiceMonth", "InvoiceMonthName"])["TotalValue"].sum().reset_index()
plt.figure(figsize=(12, 4))
plt.title("Total Revenue by Month of the Year")
sns.barplot(data=revenue_month, x="InvoiceMonthName", y="TotalValue")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.show()

# Revenue by Day of the Week
# revenue_day = df.groupby(["InvoiceWeekOfYear", "InvoiceDayOfWeek", "InvoiceDayName"])["TotalValue"].sum().reset_index()
revenue_day = df.groupby(["InvoiceDayOfWeek", "InvoiceDayName"])["TotalValue"].sum().reset_index()
revenue_day.groupby(["InvoiceDayOfWeek", "InvoiceDayName"])["TotalValue"].mean().reset_index()
plt.title("Average Revenue by Month of the Year")
sns.barplot(data=revenue_day, x="InvoiceDayName", y="TotalValue")
plt.xlabel("Day of Week")
plt.ylabel("Total Revenue")
plt.show()

# Top 10 Customers by Revenue
top_10_customers = df_customers["MonetaryValue"].sort_values(ascending=False).head(10)

# Customers RFM
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Features Distribution")
sns.histplot(df_customers["Recency"], bins=50, ax=ax[0], kde=True)
sns.histplot(df_customers["Frequency"], bins=50, ax=ax[1], kde=True)
sns.histplot(df_customers["MonetaryValue"], bins=50, ax=ax[2], kde=True)
fig.show()

# Customer Segmentation (Clustering)
# X is our data
X = df_customers
# defining some feature transformers
boxcox_t = PowerTransformer(method="box-cox")
log_t = FunctionTransformer(func=np.log, inverse_func=np.exp)
sqrt_t = FunctionTransformer(func=np.sqrt, inverse_func=lambda x : x ** 2)
cbrt_t = FunctionTransformer(func=np.cbrt, inverse_func=lambda x : x ** 3)

X_boxcox = boxcox_t.fit_transform(X)
X_log = log_t.fit_transform(X).values
X_sqrt = sqrt_t.fit_transform(X).values
X_cbrt = cbrt_t.fit_transform(X).values

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Box-Cox Transformation")
for i in range(3):
    ax = axes[i]
    sns.distplot(X_boxcox[:, i], fit=norm, kde=False, ax=ax)
    ax.set_xlabel(df_customers.columns[i])

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Log Transformation")
for i in range(3):
    ax = axes[i]
    sns.distplot(X_log[:, i], fit=norm, kde=False, ax=ax)
    ax.set_xlabel(df_customers.columns[i])

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Squared Root Transformation")
for i in range(3):
    ax = axes[i]
    sns.distplot(X_sqrt[:, i], fit=norm, kde=False, ax=ax)
    ax.set_xlabel(df_customers.columns[i])

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Cubic Root Transformation")
for i in range(3):
    ax = axes[i]
    sns.distplot(X_cbrt[:, i], fit=norm, kde=False, ax=ax)
    ax.set_xlabel(df_customers.columns[i])

standard_scaler = StandardScaler()

transformers = ColumnTransformer(
    [
        ("boxcox", boxcox_t, ["Frequency"]),
        ("cbrt", cbrt_t, ["Recency", "MonetaryValue"])
    ],
    remainder="passthrough"
)

preprocessing = Pipeline([
    ("transformers", transformers),
    ("scaler", standard_scaler),
])

kmeans = KMeans(random_state=seed)

clusterer = Pipeline([
    ("preprocessing", preprocessing),
    ("kmeans", kmeans)
])

# Finding Number of Clusters
n_clusters = np.arange(2, 15)
inertia = []
for k in n_clusters:
    kmeans = clusterer.steps[-1][1]
    kmeans.set_params(n_clusters=k)
    clusterer.fit(X)
    inertia.append(kmeans.inertia_)

plt.title("KMeans Inertia")
plt.plot(n_clusters, inertia, color="green", marker="o")
plt.show()

n_clusters = np.arange(2, 6)


def plot_silhouettes(X, labels, ax):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    ax.set_title(f"KMeans Silhouette (k={n_clusters})")

    avg_silhouette = silhouette_score(X, labels, random_state=seed)
    clusters_silhouettes = silhouette_samples(X, labels)

    y_lower = 10
    for i in (unique_labels):
        cluster_silhouettes = np.sort(clusters_silhouettes[labels == i])
        cluster_size = len(cluster_silhouettes)

        y_upper = y_lower + cluster_size
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouettes,
        )

        y_lower = y_upper + 10

    ax.axvline(avg_silhouette, color="red", linestyle="--")


n_cols = 2
n_rows = math.ceil(len(n_clusters) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 8))
for i, k in enumerate(n_clusters):
    ax = axes[i // n_cols][i % n_cols]
    kmeans = clusterer.steps[-1][1]
    kmeans.set_params(n_clusters=k)
    labels = clusterer.fit_predict(X)
    X_transf = clusterer.transform(X)
    plot_silhouettes(X_transf, labels, ax)

# removing unused axes
i += 1
while i < (n_rows * n_cols):
    ax = axes[i // n_cols][i % n_cols]
    ax.remove()

kmeans = clusterer.steps[-1][1]
kmeans.set_params(n_clusters=2)
labels = clusterer.fit_predict(X)
df_customers["Label"] = labels
df_customers.head()

def Q1(x):
    return np.percentile(x, 25)

def Q3(x):
    return np.percentile(x, 75)

df_customers.groupby("Label")[["Recency", "Frequency", "MonetaryValue"]].agg({
    "Recency": [np.mean, Q1, Q3],
    "Frequency": [np.mean, Q1, Q3],
    "MonetaryValue": [np.mean, Q1, Q3]
})

X_transf = preprocessing.fit_transform(X)

fig = px.scatter_3d(df_customers, x="Recency", y="Frequency", z="MonetaryValue", color=df_customers["Label"])
fig.show()