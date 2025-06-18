import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from scipy.stats import zscore

# 1. Nettoyage des données
df = pd.read_csv("advertising.csv")
print("Aperçu des données :")
print(df.head())

print("\nValeurs manquantes :")
print(df.isnull().sum())
df = df.dropna()

# Détection et suppression des outliers (z-score > 3)

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
cols = ['tv', 'radio', 'journaux', 'ventes']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, col in zip(axes, cols):
    # tracer l'histogramme
    counts, bins, patches = ax.hist(df[col], bins=20, edgecolor='black')
    
    # titre et labels
    ax.set_title(col.capitalize())
    ax.set_xlabel(f"{col} (M€)" if col != 'ventes' else "Ventes (unités)")
    ax.set_ylabel("Nombre d'observations")
    
    # annotation des barres
    for count, patch in zip(counts, patches):
        # label au sommet de la barre
        x = patch.get_x() + patch.get_width()/2
        y = patch.get_height()
        ax.text(x, y + 0.5,  
                f"{int(count)}",
                ha='center', va='bottom', fontsize=8)

plt.suptitle("Histogrammes des variables", y=0.98, fontsize=16)
plt.tight_layout()
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
g = sns.pairplot(df, x_vars=['tv', 'radio', 'journaux'], y_vars='ventes', kind='reg', height=4, aspect=1.2)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Relations entre les budgets publicitaires et les ventes", fontsize=16)

for ax in g.axes.flat:
    ax.set_xlabel(f"{ax.get_xlabel()} (M€)" if ax.get_xlabel() != 'ventes' else "Ventes (unités)")
    ax.set_ylabel("Ventes (unités)")
plt.show()


# On ajuste le modèle pour prédir les ventes à partir des 3 budgets publicitaires
# 3. Régressions
X = df[['tv', 'radio', 'journaux']]
y = df['ventes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# On divise les données en 80 % pour l'entraînement et 20% pour le test

# Preparation des modèles:

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



# Visualisation pour chaque modèle
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label="Prédictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Prédiction parfaite")
plt.xlabel("Ventes réelles  (M€) ")
plt.ylabel("Ventes prédites (M€) ")
plt.legend()
plt.title("Prédictions vs Réalité - Régression Linéaire")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_ridge, color='green', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Ventes réelles (M€)")
plt.ylabel("Ventes prédites (M€)")
plt.title("Prédictions vs Réalité - Ridge")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_lasso, color='red', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Ventes réelles (M€)")
plt.ylabel("Ventes prédites (M€)")
plt.title("Prédictions vs Réalité - Lasso")
plt.show()



# Fonction qui permet de calculer deux métriques 
# 4. Comparaison des performances
# y_true = valeurs réélles 
# y_perd = valeur du modèle
def print_perf(y_true, y_pred, name):
    # Calcul du RMSE & MAPE grâce à la bibliothèque sklearn : is a Python module integrating classical machine
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{name} - RMSE : {rmse:.2f} | MAPE : {mape:.2f}")
    
    
#Compare les prédictions des trois modèles sur le jeu de test :
print("\nComparaison des performances :")
print_perf(y_test, y_pred, "Régression linéaire")
print_perf(y_test, y_pred_ridge, "Ridge")
print_perf(y_test, y_pred_lasso, "Lasso")

# Explication du RMSE et du MAPE
print("\nQu'est-ce que le RMSE ?")
print("Le RMSE (Root Mean Squared Error) mesure l'erreur moyenne entre les valeurs réelles et les prédictions.")
print("Il est exprimé dans la même unité que la variable cible ('ventes' ici).")
print("Un RMSE de 1.66 est relativement bon si les ventes varient entre 1.6 et 27.0, car cela indique une erreur faible par rapport à l'échelle des données.")

print("\nQu'est-ce que le MAPE ?")
print("Le MAPE (Mean Absolute Percentage Error) mesure l'erreur moyenne en pourcentage entre les valeurs réelles et les prédictions.")
print("Un MAPE de 0.12 (12%) signifie que les prédictions sont en moyenne à 12% d'écart des valeurs réelles.")
print("C'est une bonne performance dans le contexte de prédiction des ventes.")

# Commentaire sur les performances
# RMSE de 1.66 : Bon, car il est faible par rapport à l'échelle des ventes (1.6 à 27.0).
# MAPE de 0.12 : Bon, car une erreur moyenne de 12% est acceptable dans ce type de modèle.

# Visualisation comparative des trois modèles
plt.figure(figsize=(8, 5))
# plt.scatter()	Compare graphiquement les vraies valeurs et les prédictions
plt.scatter(y_test, y_pred, label="Linéaire", alpha=0.7)
plt.scatter(y_test, y_pred_ridge, label="Ridge", alpha=0.7)
plt.scatter(y_test, y_pred_lasso, label="Lasso", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Prédiction parfaite")
plt.xlabel("Ventes réelles en M€")
plt.ylabel("Ventes prédites en ²M€")
plt.legend()
plt.title("Comparaison des modèles : Prédictions vs Réalité")
plt.show()


# Crée un nuage de points des prédictions vs les valeurs réelles pour chaque modèle.

# La ligne en pointillé représente une prédiction parfaite (y_pred = y_true).

# Plus les points sont proches de cette ligne, plus les prédictions sont précises.
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
