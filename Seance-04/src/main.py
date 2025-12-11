print('Bienvenue dans le cours d\'analyse de données en géographie !')

import numpy
import pandas as pd
import geopandas as gdp

data = pd.DataFrame({'A': [1, 2, 3]})
print(data)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (rv_discrete, randint, binom, poisson, zipf,
                         norm, lognorm, uniform, chi2, pareto)
import os

# Création dossier images
os.makedirs("img", exist_ok=True)


# 1) Fonctions utilitaires


def plot_distribution(x, y, title, filename, xlabel="Valeurs", ylabel="Probabilité"):
    """Affiche et sauvegarde une distribution discrète ou continue."""
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(f"img/{filename}.png")
    plt.close()

def compute_stats(values):
    """Retourne moyenne et écart-type d’un ensemble de valeurs."""
    return np.mean(values), np.std(values)


# 2) DISTRIBUTIONS DISCRÈTES


# 2.1 Loi de Dirac (tout le poids en un point, ex: x0 = 5)
dirac_x0 = 5
x_dirac = np.arange(0, 11)
y_dirac = np.where(x_dirac == dirac_x0, 1, 0)
plot_distribution(x_dirac, y_dirac, "Loi de Dirac (δ₅)", "dirac")

# 2.2 Uniforme discrète entre 1 et 10
x_uni = np.arange(1, 11)
y_uni = randint(1, 11).pmf(x_uni)
plot_distribution(x_uni, y_uni, "Loi uniforme discrète (1–10)", "uniforme_discrete")

# 2.3 Binomiale (n=20, p=0.5)
n, p = 20, 0.5
x_binom = np.arange(0, n+1)
y_binom = binom(n, p).pmf(x_binom)
plot_distribution(x_binom, y_binom, "Loi binomiale", "binomiale")

# 2.4 Poisson discrète (λ=4)
lam = 4
x_pois = np.arange(0, 20)
y_pois = poisson(lam).pmf(x_pois)
plot_distribution(x_pois, y_pois, "Loi de Poisson (discrète)", "poisson_discrete")

# 2.5 Zipf-Mandelbrot (approx Zipf)
a = 2
x_zipf = np.arange(1, 50)
y_zipf = zipf(a).pmf(x_zipf)
plot_distribution(x_zipf, y_zipf, "Loi de Zipf-Mandelbrot", "zipf_mandelbrot")


# 3) DISTRIBUTIONS CONTINUES


# Création d’un axe commun
x = np.linspace(-5, 20, 500)

# 3.1 Poisson ? (Poisson est **discrète**, mais on montre la courbe lissée)
y_pois_c = poisson(lam).pmf(np.floor(x))
plot_distribution(x, y_pois_c, "Poisson (visualisation continue)", "poisson_continue")

# 3.2 Normale
mu, sigma = 0, 1
y_norm = norm(mu, sigma).pdf(x)
plot_distribution(x, y_norm, "Loi normale N(0,1)", "normale")

# 3.3 Log-normale
y_logn = lognorm(s=0.5).pdf(x)
plot_distribution(x, y_logn, "Loi log-normale", "lognormale")

# 3.4 Uniforme continue entre 0 et 10
x_uni_c = np.linspace(0, 10, 500)
y_uni_c = uniform(0, 10).pdf(x_uni_c)
plot_distribution(x_uni_c, y_uni_c, "Uniforme continue (0–10)", "uniforme_continue")

# 3.5 Chi-deux (k=4)
k = 4
x_chi = np.linspace(0, 20, 500)
y_chi = chi2(k).pdf(x_chi)
plot_distribution(x_chi, y_chi, "Chi-deux (k=4)", "chi2")

# 3.6 Pareto (α = 3)
x_par = np.linspace(1, 10, 500)
y_par = pareto(b=3).pdf(x_par)
plot_distribution(x_par, y_par, "Loi de Pareto", "pareto")


# 4) Calcul moyenne + écart type


distributions = {
    "Dirac": x_dirac,
    "Uniforme discrète": randint(1, 11).rvs(10_000),
    "Binomiale": binom(n, p).rvs(10_000),
    "Poisson discrète": poisson(lam).rvs(10_000),
    "Zipf": zipf(a).rvs(10_000),
    "Normale": norm(mu, sigma).rvs(10_000),
    "Lognormale": lognorm(s=0.5).rvs(10_000),
    "Uniforme continue": uniform(0, 10).rvs(10_000),
    "Chi2": chi2(k).rvs(10_000),
    "Pareto": pareto(b=3).rvs(10_000)
}

print("\n=== Moyenne & Écart-type des distributions ===\n")

for name, sample in distributions.items():
    mean, std = compute_stats(sample)
    print(f"{name:20s} | Moyenne = {mean:.3f} | Écart-type = {std:.3f}")

