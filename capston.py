import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("penguins")

print("First Five Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

print("\nCleaned Dataset:")
print(df.head())

print("\nPenguin Species Count:")
print(df["species"].value_counts())

print("\nAverage Measurements:")
print(df.groupby("species")[
    ["bill_length_mm", "bill_depth_mm",
     "flipper_length_mm", "body_mass_g"]
].mean())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="species", hue="species")
plt.title("Number of Penguins by Species")
plt.xlabel("Penguin Species")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="species",
                  y="body_mass_g", hue="species")
plt.title("Average Body Mass by Species")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df,
                x="flipper_length_mm",
                y="body_mass_g",
                 hue="species")
plt.title("Flipper Length vs Body Mass")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="species",
            y="bill_length_mm", hue="species")
plt.title("Bill Length by Penguin Species")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="body_mass_g",
              hue="species", kde=True)
plt.title("Penguin Body Mass Distribution")
plt.show()

sns.pairplot(df, hue="species")
plt.suptitle("Penguin Species Comparison",
             y=1.02)
plt.show()

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Penguin Measurements Correlation")
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=df,
                              x="species",
               y="flipper_length_mm",
               hue="species")
plt.title("Flipper Length by Species")
plt.show()

print("\nFINAL PROJECT SUMMARY")

print("Total Penguins:", len(df))

print("\nSpecies:")
print(df["species"].unique())

print("\nAverage Body Mass:")
print(df.groupby("species")["body_mass_g"].mean())

print("\nProject Completed Successfully!")