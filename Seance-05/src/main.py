print('Bienvenue dans le cours d\'analyse de données en géographie !')

import numpy
import pandas as pd
import geopandas as gdp



# main.py
import os
import math
import numpy as np
import pandas as pd
from scipy import stats


# Configuration & utilitaires

DATA_DIR = "data"  # chemin relatif depuis Seance-05/src/
os.makedirs(DATA_DIR, exist_ok=True)

def ouvrirUnFichier(path):
    """Ouvre un CSV en utilisant 'with' et retourne un DataFrame pandas."""
    with open(path, "r", encoding="utf-8") as f:
        df = pd.read_csv(f)
    return df

def arrondir(val, ndigits=2):
    """Petit wrapper pour arrondir (garde float ou array)."""
    if isinstance(val, (np.ndarray, pd.Series)):
        return np.round(val, ndigits)
    else:
        return round(float(val), ndigits)

def intervalle_fluctuation(p_hat, n, z=1.96):
    """Retourne (p_minus, p_plus) pour un estimateur de fréquence p_hat et taille n."""
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat - z * se, p_hat + z * se

def intervalle_confiance_proportion(p_hat, n, z=1.96):
    """Même formule (ici on garde identique)."""
    return intervalle_fluctuation(p_hat, n, z=z)


# 1) Théorie de l'échantillonnage


print("\n=== THEORIE DE L'ECHANTILLONNAGE ===\n")

# Charger le fichier des 100 échantillons
path_ech = os.path.join(DATA_DIR, "Echantillonnage-100-Echantillons.csv")
df_samples = ouvrirUnFichier(path_ech)

# On suppose que les colonnes correspondent aux modalités 'Pour', 'Contre', 'Sans opinion'
print("Colonnes détectées dans Echantillonnage:", list(df_samples.columns))

# 1.a Calculer la moyenne (par colonne) sur les 100 échantillons
means_counts = df_samples.mean(axis=0)  # moyenne des comptes par modalité
means_counts_rounded = means_counts.round(0).astype(int)  # arrondi sans décimale
print("\nMoyennes (counts) par modalité (arrondi entier) :")
print(means_counts_rounded)

# 1.b Calculer fréquences issues des moyennes
sum_means = means_counts.sum()  # somme des trois moyennes (float)
frequences_means = (means_counts / sum_means).round(2)
print("\nFréquences (moyennant les échantillons) (arrondies 2 décimales):")
print(frequences_means)

# 1.c Fréquences population mère (données fournies)
pop_tot = 2185
pop_pour, pop_contre, pop_sans = 852, 911, 422
pop_freqs = pd.Series({
    df_samples.columns[0]: round(pop_pour / pop_tot, 2),
    df_samples.columns[1]: round(pop_contre / pop_tot, 2),
    df_samples.columns[2]: round(pop_sans / pop_tot, 2),
})
print("\nFréquences population mère (arrondies 2 décimales):")
print(pop_freqs)

# 1.d Intervalle de fluctuation (95%) pour chaque fréquence (utilise N = somme des moyennes, arrondie)
N = int(round(sum_means))
print(f"\nTaille utilisée pour l'intervalle (N = somme des moyennes) : {N}")
fluct_intervals = {}
for col in df_samples.columns:
    p_hat = float(means_counts[col] / sum_means)
    low, high = intervalle_fluctuation(p_hat, N, z=1.96)
    fluct_intervals[col] = (round(low, 4), round(high, 4))

print("\nIntervalle de fluctuation (95%) pour chaque modalité (valeurs proportionnelles) :")
for k, (lo, hi) in fluct_intervals.items():
    print(f" - {k}: [{lo:.4f} , {hi:.4f}] (p_hat = {round(float(means_counts[k]/sum_means),4)})")

# Sauvegarde résultats
out1 = pd.DataFrame({
    "mean_counts": means_counts.round(2),
    "mean_counts_rounded": means_counts_rounded,
    "freq_from_means": frequences_means,
    "pop_freq": pop_freqs
})
out1.to_csv(os.path.join(DATA_DIR, "echantillonnage_moyennes_frequences.csv"), index=True)


# 2) Théorie de l'estimation


print("\n=== THEORIE DE L'ESTIMATION ===\n")

# 2.a Prendre le premier échantillon (ligne 0)
first_sample = df_samples.iloc[0]
print("Premier échantillon (comptes):")
print(first_sample)

# Convertir en list (exercice demandé)
first_sample_list = list(first_sample.astype(int))
total_first = sum(first_sample_list)
print(f"Somme (taille) du premier échantillon : {total_first}")

# 2.b Calculer fréquences de cet échantillon (arrondir si nécessaire)
freq_first = [round(c / total_first, 4) for c in first_sample_list]
print("\nFréquences (premier échantillon) :")
for col, f in zip(df_samples.columns, freq_first):
    print(f" - {col}: {f}")

# 2.c Intervalle de confiance (par proportion) pour chaque modalité sur cet échantillon
ci_first = {}
zC = 1.96
for col, count in zip(df_samples.columns, first_sample_list):
    p_hat = count / total_first
    low, high = intervalle_confiance_proportion(p_hat, total_first, z=zC)
    ci_first[col] = (round(low, 4), round(high, 4))

print("\nIntervalle de confiance (95%) pour les fréquences du 1er échantillon :")
for k, (lo, hi) in ci_first.items():
    print(f" - {k}: [{lo:.4f} , {hi:.4f}] (p_hat = {round(first_sample[k]/total_first,4)})")

# (Optionnel) comparer ces intervalles avec l'intervalle de fluctuation calcule précédemment
print("\nComparaisons rapides (p_hat du 1er échantillon vs p_hat moyenné sur 100 échantillons vs pop):")
for col in df_samples.columns:
    p1 = round(first_sample[col]/total_first, 4)
    p_mean = round(float(means_counts[col]/sum_means), 4)
    p_pop = round(float({'Pour':pop_pour,'Contre':pop_contre,'Sans opinion':pop_sans}.get(col, 0) / pop_tot), 4)
    print(f" - {col}: p1={p1} | p_mean={p_mean} | p_pop={p_pop}")


# 3) Théorie de la décision

print("\n=== THEORIE DE LA DECISION ===\n")

# Charger les deux fichiers de test
path_t1 = os.path.join(DATA_DIR, "Loi-normale-Test-1.csv")
path_t2 = os.path.join(DATA_DIR, "Loi-normale-Test-2.csv")

df_t1 = ouvrirUnFichier(path_t1)
df_t2 = ouvrirUnFichier(path_t2)

# On suppose que chaque fichier contient une colonne de valeurs (si plusieurs, on prend la première)
serie1 = df_t1.iloc[:, 0].dropna().astype(float).values
serie2 = df_t2.iloc[:, 0].dropna().astype(float).values

print(f"Loi-normale-Test-1 : n = {len(serie1)} valeurs")
print(f"Loi-normale-Test-2 : n = {len(serie2)} valeurs")

# 3.a Test de Shapiro-Wilk
sh1 = stats.shapiro(serie1)
sh2 = stats.shapiro(serie2)

print("\nTest de Shapiro-Wilk (statistique, p-value) :")
print(" - Test 1:", sh1)
print(" - Test 2:", sh2)

alpha = 0.05
is_normal_1 = sh1.pvalue > alpha
is_normal_2 = sh2.pvalue > alpha

print("\nInterprétation (avec alpha = 0.05):")
print(f" - Série 1 normale ? {'OUI' if is_normal_1 else 'NON'} (p = {sh1.pvalue:.4f})")
print(f" - Série 2 normale ? {'OUI' if is_normal_2 else 'NON'} (p = {sh2.pvalue:.4f})")

# 3.b Bonus : pour la série NON normale, tenter d'identifier une loi candidate
# On testera plusieurs lois : normal, lognormale, expon, chi2, pareto
def best_fit_distribution(data):
    """Fit several continuous distributions and return the best by KS-statistic (lower = better)."""
    candidates = {
        "norm": stats.norm,
        "lognorm": stats.lognorm,
        "expon": stats.expon,
        "chi2": stats.chi2,
        "pareto": stats.pareto
    }
    results = {}
    for name, dist in candidates.items():
        try:
            # fit parameters
            params = dist.fit(data)
            # For kstest we need a CDF callable; use dist.cdf with fitted params
            # Build lambda that gives cdf at x with params
            ks_stat, ks_p = stats.kstest(data, name, args=params)
            results[name] = {"ks_stat": ks_stat, "ks_p": ks_p, "params": params}
        except Exception as e:
            results[name] = {"error": str(e)}
    # pick best by ks_stat (lowest)
    best = None
    best_name = None
    for k, v in results.items():
        if "ks_stat" in v:
            if best is None or v["ks_stat"] < best:
                best = v["ks_stat"]
                best_name = k
    return best_name, results

# Déterminer la série non normale (si les deux normales, on indiquera les deux; si aucune, on tentera les deux)
non_normals = []
if not is_normal_1:
    non_normals.append(("Test-1", serie1))
if not is_normal_2:
    non_normals.append(("Test-2", serie2))

if len(non_normals) == 0:
    print("\nLes deux séries sont normales selon Shapiro-Wilk. Aucun ajustement complémentaire nécessaire.")
else:
    for label, serie in non_normals:
        print(f"\nAnalyse d'ajustement pour {label} (série non normale) :")
        best_name, fit_results = best_fit_distribution(serie)
        if best_name is None:
            print(" - Aucun ajustement n'a pu être calculé.")
        else:
            br = fit_results[best_name]
            print(f" - Meilleure loi candidate selon KS (stat faible) : {best_name}")
            print(f"   ks_stat = {br['ks_stat']:.4f}, ks_p = {br['ks_p']:.4f}")
            print(f"   params (fitted) = {br['params']}")

# Sauvegarde tests shapiro
sh_df = pd.DataFrame({
    "test": ["Loi-normale-Test-1", "Loi-normale-Test-2"],
    "shapiro_stat": [sh1.statistic, sh2.statistic],
    "pvalue": [sh1.pvalue, sh2.pvalue],
    "is_normal": [is_normal_1, is_normal_2]
})
sh_df.to_csv(os.path.join(DATA_DIR, "shapiro_results.csv"), index=False)

print("\n=== FIN DU SCRIPT ===\n")
print("Les résultats importants sont sauvegardés dans le dossier 'data' :")
print(" - echantillonnage_moyennes_frequences.csv")
print(" - shapiro_results.csv")
print("\nRegarde les impressions au-dessus pour l'interprétation.\n")
