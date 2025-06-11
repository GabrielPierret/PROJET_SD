import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

# 1. Nettoyage des données
df = pd.read_csv("advertising.csv")
print("Aperçu des données :")
print(df.head())

print("\nValeurs manquantes :")
print(df.isnull().sum())
df = df.dropna()

# Détection et suppression des outliers (z-score > 3)
from scipy.stats import zscore
z_scores = np.abs(zscore(df[['tv', 'radio', 'journaux', 'ventes']]))
df = df[(z_scores < 3).all(axis=1)]
print(f"\nDonnées après suppression des outliers : {df.shape[0]} lignes")

# 2. Analyse exploratoire
print("\nStatistiques descriptives :")
print(df.describe())

#Interprétation des statistiques :
# - Les ventes varient de 1.6 à 27.2, avec une moyenne de 14.0.
# - Le Budget varie fortment exemple , tv( de 0.7 à 293.6)

# Histogrammes
df.hist(bins=20, figsize=(10, 6))
plt.suptitle("Histogrammes des variables")
plt.show()

# Corrélations
print("\nCorrélation entre les variables :")
print(df.corr())
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()

# Interprétation de la corrélation :
# - 'tv' a la plus forte corrélation avec 'ventes' (0.78), suivie de 'radio' (0.35) et 'journaux' (0.22).
# - Cela suggère que les dépenses publicitaires à la télévision ont le plus grand impact sur les ventes.



# Pairplot avec titre bien visible
g = sns.pairplot(df, x_vars=['tv', 'radio', 'journaux'], y_vars='ventes', kind='reg')
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Relations entre les budgets publicitaires et les ventes", fontsize=16)
plt.show()

# Interprétation du pairplot :
# - La relation entre 'tv' et 'ventes' est clairement linéaire, indiquant que plus le budget TV est élevé, plus les ventes augmentent.
# - La relation entre 'radio' et 'ventes' semble moins claire, mais il semble que plus le budget radio est élevé, plus les ventes augmentent.
# - La relation entre 'journaux' et 'ventes' est la moins marquée, indiquant que les dépenses publicitaires dans les journaux ont le moins d'impact sur les ventes.



# On ajuste le modèle pour prédir les ventes a partir des 3 budgets publicitaires
# 3. Régressions
X = df[['tv', 'radio', 'journaux']]
y = df['ventes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle Linéaire
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Modèle Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Modèle Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# 4. Comparaison des performances
def print_perf(y_true, y_pred, name):
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{name} - RMSE : {rmse:.2f} | MAPE : {mape:.2f}")

print("\nComparaison des performances :")
print_perf(y_test, y_pred, "Régression linéaire")
print_perf(y_test, y_pred_ridge, "Ridge")
print_perf(y_test, y_pred_lasso, "Lasso")

# Visualisation individuelle pour chaque modèle
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label="Prédictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Prédiction parfaite")
plt.xlabel("Ventes réelles")
plt.ylabel("Ventes prédites")
plt.legend()
plt.title("Prédictions vs Réalité - Régression Linéaire")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_ridge, color='green', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Ventes réelles")
plt.ylabel("Ventes prédites")
plt.title("Prédictions vs Réalité - Ridge")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_lasso, color='red', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Ventes réelles")
plt.ylabel("Ventes prédites")
plt.title("Prédictions vs Réalité - Lasso")
plt.show()

# Visualisation comparative des trois modèles
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, label="Linéaire", alpha=0.7)
plt.scatter(y_test, y_pred_ridge, label="Ridge", alpha=0.7)
plt.scatter(y_test, y_pred_lasso, label="Lasso", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Prédiction parfaite")
plt.xlabel("Ventes réelles")
plt.ylabel("Ventes prédites")
plt.legend()
plt.title("Comparaison des modèles : Prédictions vs Réalité")
plt.show()

# 5. Interprétation des coefficients
print("\nCoefficients du modèle linéaire :")
for name, coef in zip(X.columns, reg.coef_):
    print(f"{name} : {coef:.4f}")

print("\nCoefficients du modèle Ridge :")
for name, coef in zip(X.columns, ridge.coef_):
    print(f"{name} : {coef:.4f}")

print("\nCoefficients du modèle Lasso :")
for name, coef in zip(X.columns, lasso.coef_):
    print(f"{name} : {coef:.4f}")

# Visualisation des coefficients
coefs = pd.DataFrame({
    'Linéaire': reg.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_
}, index=X.columns)
coefs.plot(kind='bar')
plt.title("Comparaison des coefficients des modèles")
plt.ylabel("Coefficient")
plt.show()

# Analyse utile : importance des variables
print("\nAnalyse :")
print("Les coefficients montrent l'importance de chaque budget sur les ventes.")
print("Un coefficient proche de zéro (Lasso) indique une variable peu utile.")
print("La corrélation et les histogrammes montrent que 'tv' a le plus d'impact sur les ventes.")
print("Dans ce jeu de données, les modèles Linéaire, Ridge et Lasso donnent des résultats identiques.")
print("Cela montre que la régularisation n'apporte rien ici car les variables inutiles sont déjà peu influentes.")
print("La régression linéaire simple est donc suffisante pour expliquer les ventes.")