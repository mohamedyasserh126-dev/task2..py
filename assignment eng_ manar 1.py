# =========================================================
# EDA على داتا Iris
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set(style="whitegrid")

# تحميل الداتا
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species"] = df["species"].map(dict(enumerate(iris.target_names)))

print("شكل الداتا:", df.shape)
print(df.head())
df.info()
print(df.describe())

# فحص القيم الناقصة والمكررة وتوزيع الفئات
print(df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())
print(df["species"].value_counts())

# الرسومات
sns.countplot(x="species", data=df); plt.show()
df.hist(figsize=(10,8)); plt.show()
sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="species", data=df); plt.show()
sns.boxplot(data=df.drop("species", axis=1)); plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm"); plt.show()

# تجهيز البيانات للموديل
df["species_encoded"] = LabelEncoder().fit_transform(df["species"])
scaled_features = StandardScaler().fit_transform(df.iloc[:, :-2])
print("عينة من البيانات بعد ال Scaling:\n", scaled_features[:5])

print("خلصنا EDA والتجهيز ✅")