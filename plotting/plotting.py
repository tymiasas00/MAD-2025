import matplotlib.pyplot as plt
import seaborn as sns
from data.data_analysis import movies_data

# Lista cech numerycznych do analizy
numeric_features = [
    'audienceScore', 'tomatoMeter', 'runtimeMinutes', 'scoreSentiment',
    'mean_originalScore'
]

# Histogramy (rozkład liczbowy)
for col in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=movies_data, x=col, bins=30, kde=False, stat='percent')
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Procent')
    plt.tight_layout()
    plt.savefig(f"histogram_{col}.png")
    plt.close()

# Boxploty (dla outlierów)
for col in numeric_features:
    plt.figure(figsize=(8, 2.5))
    sns.boxplot(x=movies_data[col])
    plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}.png")
    plt.close()

# Scatterploty wybranych par zmiennych
scatter_pairs = [
    ('audienceScore', 'tomatoMeter'),
    ('runtimeMinutes', 'mean_originalScore'),
    ('scoreSentiment', 'audienceScore')
]

for x, y in scatter_pairs:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=movies_data, x=x, y=y, alpha=0.5)
    plt.title(f'{x} vs {y}')
    plt.tight_layout()
    plt.savefig(f"scatter_{x}_vs_{y}.png")
    plt.close()


# Histogramy (rozkład liczbowy)
for col in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=movies_data, x=col, bins=30, kde=False, stat='percent')
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Procent')
    plt.tight_layout()
    plt.savefig(f"histogram_{col}.png")
    plt.close()

# Boxploty (dla outlierów)
for col in numeric_features:
    plt.figure(figsize=(8, 2.5))
    sns.boxplot(x=movies_data[col])
    plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}.png")
    plt.close()

# Scatterploty wybranych par zmiennych
scatter_pairs = [
    ('audienceScore', 'tomatoMeter'),
    ('runtimeMinutes', 'mean_originalScore'),
    ('scoreSentiment', 'audienceScore')
]

for x, y in scatter_pairs:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=movies_data, x=x, y=y, alpha=0.5)
    plt.title(f'{x} vs {y}')
    plt.tight_layout()
    plt.savefig(f"scatter_{x}_vs_{y}.png")
    plt.close()