# %%
import pandas as pd
import numpy as np
import joblib # sauvegarde des modèles entraînés
import pickle # sérialisation d'objets Python
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity   
from sklearn.feature_extraction.text import TfidfVectorizer


from pathlib import Path

BASE_DIR = Path.cwd().parent  

PKL_OUT = BASE_DIR / "fastapi-supermarche" / "models_pkl"
test_df = pickle.load(open(PKL_OUT / "test_df.pkl", "rb"))
hour_scores_norm = pickle.load(open(PKL_OUT / "hour_scores_norm.pkl", "rb"))
season_popularity_norm = pickle.load(open(PKL_OUT / "season_popularity_norm.pkl", "rb"))
sim_df = pickle.load(open(PKL_OUT / "sim_df.pkl", "rb"))
tfidf = joblib.load(PKL_OUT / "tfidf.pkl")
train_df = pickle.load(open(PKL_OUT / "train_df.pkl", "rb"))
products_catalog = pickle.load(open(PKL_OUT / "products_catalog.pkl", "rb"))
user_health_profile = pickle.load(open(PKL_OUT / "user_health_profile.pkl", "rb"))
product_expiration_map = pickle.load(open(PKL_OUT / "product_expiration_map.pkl", "rb"))

interactions=pd.read_csv("../data/raw/interactions.csv", encoding='latin-1')



#vectors de tous les produits existants
all_vec=tfidf.transform(products_catalog['ingredients_clean'])  # utiliser le champ deja nettoyé 
    

def compute_similarity_new_product(product_id,products_catalog,tfidf,sim_df,all_vec):
    #recupere le texte nettoyé du produit 
    new_product_text=products_catalog[products_catalog['Product_ID']==product_id]['ingredients_clean']

    if new_product_text.empty: # produit n'existe pas 
        return sim_df
    
    # vector du produit nouveau
    new_vec=tfidf.transform(new_product_text)

    #similarity cos
    sims=cosine_similarity(new_vec,all_vec)[0]

    #mettre à jour sim_df
    new_sim_series = pd.Series(sims, index=products_catalog['Product_ID'])    
    sim_df[product_id] = new_sim_series     # sauvegarde du dataframe
    sim_df.loc[product_id] = sim_df[product_id]
    sim_df.loc[product_id, product_id] = 1.0
    
    # Sauvegarde
    with open('sim_df.pkl', 'wb') as f:
        pickle.dump(sim_df, f)  
    return sim_df


#fonction score contenutest_df=pickle.load(open('test_df.pkl','rb'))
def get_score_contenu_cosinus(user_id,product_id):
    global sim_df
    if product_id not in sim_df.columns:
        sim_df=compute_similarity_new_product(product_id,products_catalog,tfidf,sim_df,all_vec) # rebuild et save pickle
    
    #récupère l'historique d'achat de user
    user_purchases=train_df[(train_df['user_id']==user_id) & #filtre sur user
                   (train_df.get('act_purchase',pd.Series(False,index=train_df.index)).astype(bool)  )    # garde seulement les achats réels  
     
                  ]['product_id'].unique().tolist()# .unique() : évite les doublons | .tolist() : liste Python
    # pas d'historique d'achat
    if not user_purchases:
        return 0
    

  # Calcule la similarité cosinus avec chaque produit acheté
    scores=[]
    for purchased_pid in user_purchases:
        if purchased_pid in sim_df.index and product_id in sim_df.columns:
            sim_score=sim_df.loc[purchased_pid,product_id] #Récupère la similarité cosinus
            scores.append(sim_score)
    if not scores:
        return 0 #aucun produit acheté dans la matrice

    return float(np.mean(scores))





# %%
#HISTORIQUE D'ACHAT PAR UTILISATEUR

# filtre les lignes où l'action était un achat réel

if 'act_purchase' not in train_df.columns:
    raise ValueError("La colonne 'act_purchase' est manquante dans train_df")

user_purchase_history = (
    train_df[train_df['act_purchase'] == 1]
    .groupby('user_id')['product_id']
    .apply(list)
    .to_dict()
)

print(f"\nUtilisateurs avec historique d'achat : {len(user_purchase_history)}/{train_df['user_id'].nunique()}")
users_with_history = {uid for uid, pids in user_purchase_history.items() if len(pids) >= 1}


def get_user_type(user_id):
    """
    Détermine le type d'utilisateur (1, 2 ou 3) selon le document :
    Type 1 : invité sans profil ni historique
    Type 2 : profil santé (sant ou non ) mais sans historique d'achat
    Type 3 : profil + historique d'achat (≥ 1 achats)
    """
    history=user_purchase_history.get(user_id,[])
    health=user_health_profile.get(user_id,None)
    # Logique de classification
    if user_id is None:
       return 1
    if len(history) >= 1:
        return 3      
    else:
        return 2
    
#test fictif pour type2 et type1 
"""
fake_user_id="fake_type1_001"
fake_health_p="Standard"
user_health_profiles[fake_user_id]=fake_health_p
user_history[fake_user_id]=[]
print(f"type est {get_user_type(fake_user_id)}")

fake_user_id="fake_type1_002"
fake_health_p=None
user_health_profiles[fake_user_id]=fake_health_p
user_history[fake_user_id]=[]
print(f"type est {get_user_type(fake_user_id)}")
"""

 


# %%
def select_with_constraints(df, n, max_per_cat=2):
    if df.empty: return df
    selected = []
    used_names = set()
    cat_count = {}

    for _, row in df.iterrows():
        name = row['Name']        
        cat  = row['Category']    

        if name in used_names:
            continue

        # 2. limit  category
        if cat_count.get(cat, 0) >= max_per_cat:
            continue

        selected.append(row)
        used_names.add(name)
        cat_count[cat] = cat_count.get(cat, 0) + 1

        if len(selected) == n:
            break

    return pd.DataFrame(selected).reset_index(drop=True)

# %%
# --- ETAPE : Data Integrity Check

if 'act_purchase' not in train_df.columns:
    print("Column 'act_purchase' not found. Checking for One-Hot version...")
    for col in train_df.columns:
        if 'act_purchase' in col or 'action_purchase' in col:
            train_df['act_purchase'] = train_df[col]
            break

if 'hour' not in train_df.columns or 'season' not in train_df.columns:
    print("Column 'hour' or 'season' are not found. Checking for One-Hot version...")

    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    train_df['hour'] = interactions.loc[train_df.index, 'timestamp'].dt.hour
    train_df['season'] = interactions.loc[train_df.index, 'season']

print("Etape : Data is ready for Modeling.")

# %% [markdown]
# fonction recommandation principale 

# %%
def recommander(user_id,n=10,current_hour=None,current_season=None,force_user_type=None,session_data=None):
 import sys
 import os
 sys.path.append(os.path.abspath(".."))

 try:
        from src_sys.preprocessing1 import (
            prefiltrage_sante, 
            get_dynamic_d_i, 
            postfiltrage_sante
        )
 except ImportError:
  print("Note: Using local functions due to import path issue.")

  # ── Étape 1 : Détecter le type d'utilisateur ────────────────
 user_type=force_user_type if force_user_type is not None else get_user_type(user_id)
 
 #profil de santé 
 hp=user_health_profile.get(user_id,'Standard')

   # ── Étape 2 : Heure et saison actuelles ─────────────────────
 if current_hour is None:
  # si pas d'heure fournie → utilise la médiane des heures du train
   current_hour = int(pd.to_numeric(train_df['hour'], errors='coerce').median()) 
 else:
   current_hour = current_hour
 # si pas de saison fournie → utilise la saison la plus fréquente dans le train
 if current_season is None:
   current_season = train_df['season'].mode()[0].lower()
 else:
   current_season = current_season.lower() # S'assurer que c'est en minuscule
 
  # ── Étape 3 : Score contextuel pour cette heure et saison ───
 S_i_now={} #dict pour stocker chaque categorie d'apres la saison actuelle
 for cat in products_catalog['Category'].unique():    # itère sur toutes les catégories du catalogue
   # récupère le score saisonnier pour (saison actuelle, catégorie)
    S_i_now[cat]=season_popularity_norm.get((current_season,cat),0.5)
 
  # ── Étape 4 : Récupérer le catalogue des produits candidats ─
 candidates = products_catalog.drop_duplicates(subset='Product_ID').copy()
 if 'stock' in candidates.columns:
    candidates = candidates[candidates['stock'] > 0]
 elif 'is_available' in candidates.columns:
    candidates = candidates[candidates['is_available'] == True]

 if candidates.empty:
    print("[WARNING] Aucun produit en stock disponible.")
    return pd.DataFrame()
  # ── Étape 5 : Pré-filtrage santé (Types 1 & 2) ──────────────

  #   AVANT le calcul des scores (sécurité alimentaire en priorité)
 
 if user_type == 2:
   candidates=prefiltrage_sante(candidates.copy(),hp)  #exclut les produits incompatibles avec le profil santé
 
  # ── Étape 6 : Exclure les produits déjà achetés (Type 3) ────
 if user_type==3:
   # ensemble des product_id déjà achetés par cet utilisateur
   already_bought=set(user_purchase_history.get(user_id,[]))
   candidates=candidates[~candidates['Product_ID'].isin(already_bought)]

   
  # ── Étape 7 : Score session pour cet utilisateur ────────────
  # On filtre sur l'ID de l'utilisateur actuel (user_id)
  
 user_actions = pd.DataFrame()

   # Cas A : On nous donne directement les données (Test manuel ou Temps Réel via API)
 if session_data is not None:
    user_actions = session_data.copy()

   # Cas b : Utilisateur connecté (Type 2) mais pas de session_id spécifique
 else :
    # On récupère les interactions historiques de l'utilisateur
    user_actions = train_df[train_df['user_id'] == user_id]
      


 if not user_actions.empty:
    user_session = user_actions.groupby('product_id')['action_score'].sum()
    total_session = user_session.sum()
    session_score = (user_session / total_session).to_dict() if total_session > 0 else {}
 else:
    session_score = {}

  # ── Étape 8 : Calcul des scores pour chaque produit candidat

 results =[]     #  liste pour accumuler les résultats
  #w_h, w_s, w_d = 0.45, 0.35, 0.20
 for _,row in candidates.iterrows():     # itère sur chaque produit candidat
   pid=row['Product_ID']
   cat = row.get('Category', 'Unknown')
       # identifiant du produit candidat
   #score contextuel:

   H_i = hour_scores_norm.get((current_hour, cat), 0.5)
      #saison:

   S_i=S_i_now.get(cat,0.5)
      #s_DLC:
      # 1. Récupération de la date
   exp_date_raw = product_expiration_map.get(pid, None)

   if exp_date_raw is None or (not isinstance(exp_date_raw, pd.Timestamp)
                             and pd.isnull(pd.to_datetime(exp_date_raw,
                                                          errors='coerce'))):
    display_date = "N/A"
   elif isinstance(exp_date_raw, pd.Timestamp):
    display_date = exp_date_raw.strftime('%d-%m-%Y') if pd.notnull(exp_date_raw) else "N/A"
   else:
    ts = pd.to_datetime(exp_date_raw, dayfirst=False, errors='coerce')
    display_date = ts.strftime('%d-%m-%Y') if pd.notnull(ts) else "N/A"

   D_i = get_dynamic_d_i(exp_date_raw)
   if D_i == 0.0:
    continue  # skip expired products entirely

      #s_context=w_h*H_i_now+w_s*S_i+w_d*D_i

   s_context = (0.45 * H_i) + (0.30 * S_i) + (0.25* D_i)

   #score contenu:
   if user_type==3:
    s_contenu=get_score_contenu_cosinus(user_id,pid)    # calcule la similarité moyenne avec les produits achetés

   else: 
    s_contenu=0        #  pas d'historique → score contenu = 0

   #score session:
   s_session=session_score.get(pid,0)

   # Score final
   if user_type ==2 :
    s_final = round(float(np.clip((0.4 * s_session) +( 0.6 * s_context), 0, 1)), 4)
   elif user_type ==1:
    s_final = round(float(np.clip((1.0* s_context), 0, 1)), 4)

   else:
    s_final = round(float(np.clip((0.4 * s_contenu) +( 0.6 * s_context), 0, 1)),4)#type3
   #l'affichage
   results.append({
            'product_id'    : pid,
            'Name'          : row.get('Name', '?'),
            'Category'      : cat,
            'score_contenu' : round(s_contenu, 4),
            'score_session' : round(s_session, 4),
            'score_context' : round(s_context, 4),
            'score_final'   : s_final,
            'H_i'           : round(H_i, 3),
            'S_i'           : round(S_i, 3),
            'DLC_Date'      : display_date,
            'D_i'           : round(D_i, 2),
        })
   
   # trie par score final décroissant → meilleure reco en premier
 full_reco_df=pd.DataFrame(results)
 
 # Post-filtrage santé appliqué sur TOUS les candidats AVANT le mix
  # ── Garde-fou : catalogue vide après scoring ─────────────────

 if full_reco_df.empty:
    print(f"[WARNING] No candidates found for user {user_id} (type {user_type}, profile {hp})")

    return full_reco_df
 
  # ── Post-filtrage santé : Type 3 uniquement ──────────────────

 if user_type == 3:
  safe_ids = postfiltrage_sante(full_reco_df['product_id'].tolist(), hp)
  full_reco_df = full_reco_df[full_reco_df['product_id'].isin(safe_ids)]
  # After post-filter, check we still have enough results
  if len(full_reco_df) < n:
    print(f"[WARNING] Only {len(full_reco_df)} safe products found "
          f"after health post-filter for profile '{hp}' — returning all available.")
  
  # ── Tri et sélection finale avec contrainte de diversité ─────

 df_sorted = full_reco_df.sort_values('score_final', ascending=False)
 mixed_reco = select_with_constraints(df_sorted, n=n, max_per_cat=2)

 return mixed_reco

 

# %%
import pickle, joblib
from pathlib import Path

import os

# Utilise le dossier actuel du notebook
current_dir = Path(os.getcwd()) 
PKL_OUT = current_dir.parent / "fastapi-supermarche" / "models_pkl"
PKL_OUT.mkdir(parents=True, exist_ok=True)

joblib.dump(tfidf, PKL_OUT / "tfidf.pkl")
pickle.dump(sim_df, open(PKL_OUT / "sim_df.pkl", "wb"))
pickle.dump(products_catalog, open(PKL_OUT / "products_catalog.pkl", "wb"))
pickle.dump(train_df, open(PKL_OUT / "train_df.pkl", "wb"))
pickle.dump(hour_scores_norm, open(PKL_OUT / "hour_scores_norm.pkl", "wb"))
pickle.dump(season_popularity_norm, open(PKL_OUT / "season_popularity_norm.pkl", "wb"))
pickle.dump(user_health_profile, open(PKL_OUT / "user_health_profile.pkl", "wb"))
pickle.dump(product_expiration_map, open(PKL_OUT / "product_expiration_map.pkl", "wb"))

print(f" Tous les .pkl sauvegardés dans : {PKL_OUT}")

# %% [markdown]
# test
# 

# %%
# ── Test Type 3 : utilisateur avec historique ────────────────────
user_type3 = list(users_with_history)[0] if users_with_history else train_df['user_id'].iloc[0]
# ^ prend un user avec ≥ 3 achats pour tester le Type 3
print(hour_scores_norm)
print(f"\n[TYPE 3] Utilisateur : {user_type3[:8]}...")
print(f"  Profil santé : {user_health_profile.get(user_type3, 'Standard')}")
print(f"  Nb achats    : {len(user_purchase_history.get(user_type3, []))}")
reco3 = recommander(user_type3, n=10, current_hour=12, current_season='spring')
reco_22 = recommander(user_type3, n=10, current_hour=10, current_season='winter')

print(f"\n  Top 10 recommandations (Type 3 — 1h, winter) :")
print(reco3[['Name','Category','score_contenu','score_context','score_final', 'DLC_Date','D_i']].to_string(index=False))


print(f"\n  Top 10 recommandations (Type 3 — 10h, winter) :")

print(reco_22[['Name','Category','score_contenu','score_context','score_final', 'DLC_Date','D_i']].to_string(index=False))


# %%
# ── Test Type 1 : Invité avec Session Active ─────────────────────────

# 1. On définit un ID de session fictif pour notre invité

# 2. On simule des interactions pour cette session (ex: l'invité regarde des fruits)
# On récupère des IDs de produits existants dans le catalogue pour que le test soit réel


# 4. On lance la recommandation
print(f"\n[TYPE 1] Utilisateur invité (ID Session)")
reco1 = recommander(
    user_id='unknown_guest', 
    n=5, 
    current_hour=6,
    current_season='winter', 
    force_user_type=1,

)

print(f"\n  Top 5 recommandations (Type 1 — 10h, spring, Session Active) :")
print(reco1[['Name','Category','score_session','score_context','score_final','D_i']].to_string(index=False))

# %%
# ── Test Type 2 : Utilisateur identifié, sans historique d'achat ────────

# 1. Sélection d'un utilisateur réel (qui a potentiellement un profil santé)
# On prend un utilisateur au hasard dans le train_df
user_type2 = train_df[train_df['user_id'] != 'unknown_guest']['user_id'].iloc[5]

# 2. Simulation d'une session active (ce qu'il regarde en ce moment)
fake_sid_t2 = "sess_type2_test"
# On choisit un produit de la catégorie 'Snacks' ou 'Dairy' pour le test
p_session = products_catalog.iloc[20]['Product_ID'] 

session_data_t2 = pd.DataFrame([
    {
        'user_id': user_type2, 
        'session_id': fake_sid_t2, 
        'product_id': p_session, 
        'action': 'view', 
        'action_score': 1
    }
])

# 3. Lancement de la recommandation
print(f"\n[TYPE 2] Utilisateur identifié : {user_type2[:8]}...")
print(f"  Profil santé stocké : {user_health_profile.get(user_type2, 'Standard')}")

reco2 = recommander(
    user_id=user_type2, 
    n=10, 
    current_hour=2,           # 16h : Goûter / Fin de journée
    current_season='summer', 
    force_user_type=2,         # On force la logique Type 2 (Session + Contexte)
    session_data=session_data_t2
)

# 4. Affichage des résultats
print(f"\n  Top 5 recommandations (Type 2 — 16h, spring, Focus Session) :")
print(reco2[['Name','Category','score_session','score_context','score_final']].to_string(index=False))

# %%
# On cible l'utilisateur 9e81b4d0 qui a interagi avec le yaourt
target_user = "9e81b4d0-36eb-4253-af33-27e25ef71e3c"

print(f"--- Test de Recommandation Hybride ---")
print(f"Utilisateur : {target_user}")
print(f"Profil Santé : {user_health_profile.get(target_user, 'None')}")

# Lancer la recommandation
recos = recommander(target_user, n=5)

if not recos.empty:
    print("\nTop 5 Recommandations pour cet utilisateur :")
    # Affichage des scores pour vérifier la logique Yogurt A, B, C, D
    cols = ['Name', 'Category', 'score_contenu', 'score_contextuel', 'score_final']
    existing_cols = [c for c in cols if c in recos.columns]
    print(recos[existing_cols].to_string(index=False))
else:
    print("⚠ Aucun résultat trouvé pour cet utilisateur.")


