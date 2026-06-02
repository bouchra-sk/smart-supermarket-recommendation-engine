# %%
"""
evaluation_metrics.py
=====================
Métriques d'évaluation pour le système de recommandation hybride.
Couvre : Precision@K, Recall@K, NDCG@K, F1@K, Hit Rate@K,
         Coverage, Diversity (intra-list), MRR, et un rapport global.
"""

import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
PKL_OUT = Path().resolve().parent / "fastapi-supermarche"/ "models_pkl"

test_df          = pickle.load(open(PKL_OUT / "test_df.pkl",          "rb"))
train_df         = pickle.load(open(PKL_OUT / "train_df.pkl",         "rb"))
products_catalog = pickle.load(open(PKL_OUT / "products_catalog.pkl", "rb"))
sim_df           = pickle.load(open(PKL_OUT / "sim_df.pkl",           "rb"))
user_health_profile = pickle.load(open(PKL_OUT / "user_health_profile.pkl", "rb"))

# import de la fonction de recommandation principale
import sys, os
sys.path.append(os.path.abspath(".."))
from src_sys.modeling import recommander          # ajuster le chemin si besoin

# ─────────────────────────────────────────────
# 1. MÉTRIQUES DE BASE  @K
# ─────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Proportion des K premières reco qui sont pertinentes."""
    if not recommended or not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Proportion des items pertinents retrouvés dans le top-K."""
    if not recommended or not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Moyenne harmonique Precision@K et Recall@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """NDCG@K : pénalise les bonnes reco placées loin en tête."""
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    dcg  = sum(1 / np.log2(i + 2) for i, p in enumerate(top_k) if p in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """1 si au moins 1 item pertinent figure dans le top-K, sinon 0."""
    if not recommended or not relevant:
        return 0.0
    return 1.0 if set(recommended[:k]) & relevant else 0.0


def mean_reciprocal_rank(recommended: list, relevant: set) -> float:
    """MRR : 1/rang du premier item pertinent (0 si aucun)."""
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────
# 2. MÉTRIQUES SYSTÈME (Coverage & Diversity)
# ─────────────────────────────────────────────

def catalog_coverage(all_recommended_ids: list, catalog_ids: list) -> float:
    """
    Part du catalogue recommendé au moins une fois.
    → mesure si le système explore bien tout le catalogue.
    """
    return len(set(all_recommended_ids)) / len(set(catalog_ids))


def intra_list_diversity(recommended: list, sim_matrix: pd.DataFrame) -> float:
    """
    Diversité intra-liste : 1 - similarité cosinus moyenne entre toutes
    les paires de produits recommandés.
    Valeur proche de 1 → liste très diversifiée.
    """
    ids = [p for p in recommended if p in sim_matrix.index]
    if len(ids) < 2:
        return 1.0
    sims = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            s = sim_matrix.loc[ids[i], ids[j]] if ids[j] in sim_matrix.columns else 0.0
            sims.append(float(s))
    return 1.0 - np.mean(sims) if sims else 1.0


# ─────────────────────────────────────────────
# 3. CONSTRUCTION DU GROUND TRUTH DEPUIS TEST_DF
# ─────────────────────────────────────────────

def build_ground_truth(test_df: pd.DataFrame,
                        min_action_score: int = 3) -> dict:
    """
    Extrait les produits 'pertinents' pour chaque utilisateur dans le test set.
    Pertinent = action_score >= min_action_score (add_to_cart ou purchase).

    Returns
    -------
    dict  {user_id : set(product_id)}
    """
    relevant_actions = test_df[test_df['action_score'] >= min_action_score]
    ground_truth = (
        relevant_actions
        .groupby('user_id')['product_id']
        .apply(set)
        .to_dict()
    )
    return ground_truth


# ─────────────────────────────────────────────
# 4. ÉVALUATION GLOBALE SUR UN ÉCHANTILLON
# ─────────────────────────────────────────────

def evaluate_recommender(
    ground_truth: dict,
    k: int = 10,
    n_users: int = 100,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Évalue la fonction `recommander()` sur un échantillon d'utilisateurs.

    Parameters
    ----------
    ground_truth : dict  {user_id : set(product_id)}
    k            : taille de la liste recommandée à évaluer
    n_users      : nombre d'utilisateurs à évaluer (sous-ensemble pour rapidité)
    random_seed  : reproductibilité

    Returns
    -------
    pd.DataFrame  une ligne par utilisateur + ligne moyenne
    """
    ground_truth = {str(uid).replace("-", ""): v for uid, v in ground_truth.items()}

    rng       = np.random.default_rng(random_seed)
    user_ids  = list(ground_truth.keys())
    sample    = rng.choice(user_ids, size=min(n_users, len(user_ids)), replace=False)

    catalog_ids       = products_catalog['Product_ID'].tolist()
    all_recommended   = []  # pour catalog_coverage
    rows              = []

    for uid in sample:
        relevant = ground_truth[uid]

        # ── Appel au moteur de recommandation ──
        try:
            eval_hour = int(pd.to_numeric(test_df["hour"], errors="coerce").median())
            eval_season = test_df["season"].mode()[0].lower()

            reco_df = recommander(uid, n=k, current_hour=eval_hour, current_season=eval_season)
            recommended = reco_df['product_id'].tolist() if not reco_df.empty else []
        except Exception as e:
            print(f"[WARN] recommander({uid}) → {e}")
            recommended = []

        all_recommended.extend(recommended)

        # ── Calcul des métriques ──
        rows.append({
            'user_id'     : uid,
            'user_type'   : _detect_type(uid),
            'health_profile' : user_health_profile.get(uid, 'Standard'),
            'n_relevant'  : len(relevant),
            'n_recommended': len(recommended),
            f'Precision@{k}' : precision_at_k(recommended, relevant, k),
            f'Recall@{k}'    : recall_at_k(recommended, relevant, k),
            f'F1@{k}'        : f1_at_k(recommended, relevant, k),
            f'NDCG@{k}'      : ndcg_at_k(recommended, relevant, k),
            f'HitRate@{k}'   : hit_rate_at_k(recommended, relevant, k),
            'MRR'            : mean_reciprocal_rank(recommended, relevant),
            'Diversity'      : intra_list_diversity(recommended, sim_df),
        })

    results_df = pd.DataFrame(rows)

    # ── Ligne récapitulative (moyenne) ──
    mean_row = results_df.drop(columns=['user_id', 'user_type', 'health_profile']).mean().to_dict()
    mean_row['user_id']       = 'MEAN'
    mean_row['user_type']     = '-'
    mean_row['health_profile']= '-'
    results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)

    # ── Coverage catalogue ──
    cov = catalog_coverage(all_recommended, catalog_ids)
    print(f"\n{'─'*55}")
    print(f"  Catalog Coverage  : {cov:.2%}  ({len(set(all_recommended))}/{len(catalog_ids)} produits)")
    print(f"{'─'*55}")

    return results_df



# ─────────────────────────────────────────────
# 5. RAPPORT DÉTAILLÉ PAR TYPE D'UTILISATEUR
# ─────────────────────────────────────────────

def _detect_type(user_id: str) -> int:
    """Réplique la logique get_user_type() sans importer modeling."""
    if user_id is None:
        return 1
    history = train_df[
        (train_df['user_id'] == user_id) &
        (train_df.get('act_purchase', pd.Series(False, index=train_df.index)).astype(bool))
    ]['product_id'].tolist()
    return 3 if len(history) >= 1 else 2


def report_by_user_type(results_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Agrège les métriques par type d'utilisateur (1, 2, 3).
    La ligne MEAN est exclue de l'agrégation.
    """
    metric_cols = [c for c in results_df.columns
                   if any(c.startswith(m) for m in
                          ['Precision', 'Recall', 'F1', 'NDCG', 'HitRate', 'MRR', 'Diversity'])]
    df = results_df[results_df['user_id'] != 'MEAN'].copy()
    df['user_type'] = df['user_type'].astype(str)
    report = df.groupby('user_type')[metric_cols].mean().round(4)
    return report


# ─────────────────────────────────────────────
# 6. AFFICHAGE FORMATÉ
# ─────────────────────────────────────────────

def print_evaluation_report(results_df: pd.DataFrame, k: int = 10) -> None:
    mean_row = results_df[results_df['user_id'] == 'MEAN'].iloc[0]

    print("\n" + "═" * 55)
    print("   RAPPORT D'ÉVALUATION — SYSTÈME DE RECOMMANDATION")
    print("═" * 55)
    print(f"  Utilisateurs évalués : {len(results_df) - 1}")
    print(f"  K = {k}")
    print("─" * 55)

    metrics = [
        (f'Precision@{k}', "Précision"),
        (f'Recall@{k}',    "Rappel"),
        (f'F1@{k}',        "F1-Score"),
        (f'NDCG@{k}',      "NDCG"),
        (f'HitRate@{k}',   "Hit Rate"),
        ('MRR',            "MRR"),
        ('Diversity',      "Diversité intra-liste"),
    ]
    for col, label in metrics:
        val = mean_row.get(col, float('nan'))
        print(f"  {label:<28} : {val:.4f}")

    print("─" * 55)

    print("\n  Détail par type d'utilisateur :")
    type_report = report_by_user_type(results_df, k)
    print(type_report.to_string())
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────
# 7. MAIN — EXÉCUTION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    K = 10

    # ── Vérification rapide des métriques de base (exemple hardcodé) ──
    print("=== Test unitaire des métriques ===")
    rec_ex = [234, 221, 233, 246, 201]
    rel_ex = {234, 221}
    print(f"  Recommandés : {rec_ex}")
    print(f"  Pertinents  : {rel_ex}")
    print(f"  Precision@5 : {precision_at_k(rec_ex, rel_ex, 5):.2f}")
    print(f"  Recall@5    : {recall_at_k(rec_ex, rel_ex, 5):.2f}")
    print(f"  NDCG@5      : {ndcg_at_k(rec_ex, rel_ex, 5):.2f}")
    print(f"  Hit Rate@5  : {hit_rate_at_k(rec_ex, rel_ex, 5):.2f}")
    print(f"  MRR         : {mean_reciprocal_rank(rec_ex, rel_ex):.2f}")

    # ── Ground truth depuis test_df ──
    print("\nConstruction du ground truth depuis test_df …")
    gt = build_ground_truth(test_df, min_action_score=3)
    print(f"  Utilisateurs avec items pertinents : {len(gt)}")

    # ── Évaluation globale ──
    print("\nÉvaluation en cours (100 utilisateurs max) …")
    results = evaluate_recommender(gt, k=K, n_users=100)

    # ── Rapport ──
    print_evaluation_report(results, k=K)

    # ── Sauvegarde CSV ──
    out_path = Path("evaluation_results.csv")
    results.to_csv(out_path, index=False)
    print(f"  Résultats sauvegardés → {out_path.resolve()}")


