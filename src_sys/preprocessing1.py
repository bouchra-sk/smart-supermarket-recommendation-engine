# %% [markdown]
# ## Phase1 : Format propice au ML

# %% [markdown]
# Train/ Test split -découpage temporel

# %%
import pandas as pd                          # manipulation des tableaux (DataFrames)
import numpy  as np                          # calculs numériques (vecteurs, matrices)
import matplotlib.pyplot as plt              # création de graphiques
import seaborn as sns                        # graphiques statistiques avancés
import warnings                              # gestion des avertissements Python
warnings.filterwarnings('ignore')            # masque les warnings non critiques
 
from sklearn.preprocessing import LabelEncoder      # encodage ordinal (Label Encoding)
from sklearn.preprocessing  import MinMaxScaler       # normalisation dans [0, 1]
from sklearn.preprocessing  import StandardScaler     # standardisation (µ=0, σ=1)
from sklearn.feature_extraction.text import TfidfVectorizer   # vectorisation TF-IDF sur texte
from sklearn.metrics.pairwise        import cosine_similarity  # similarité cosinus entre vecteurs


# Chargement des fichiers CSV dans des DataFrames
interactions = pd.read_csv("../data/raw/interactions.csv", encoding='latin-1') # Dataset des interactions utilisateurs
products=pd.read_csv("../data/raw/produits.csv",encoding='latin-1')        # Dataset des produits

interactions.shape
products.shape

# %%
#construire un catalogue du produits complets 
# Extraction des produits uniques depuis interactions
interactions.columns = interactions.columns.str.strip()
products_catalog=(interactions[['product_id','product_name','category','ingredients']].drop_duplicates('product_id') #garde 1 seule ligne par produit unique (240 produits distincts)
                  .reset_index(drop=True)) # réinitialise l'index)

interactions['barcode'] = interactions['barcode'].astype(str).str.split('.').str[0]
products['barcode'] = products['barcode'].astype(str).str.split('.').str[0]
#


# Création d’un dictionnaire associant chaque product_id à son barcode
barcode_map=(interactions[['product_id','barcode']].drop_duplicates('product_id')
             .set_index('product_id')['barcode'] # ^ product_id devient l'index → accès direct par product_id
             .to_dict() # ^ convertit en dictionnaire {product_id → barcode}
             )
# Association de chaque produit avec son barcode via mapping
products_catalog['barcode']=products_catalog['product_id'].map(barcode_map) #associe chaque product_id à son barcode

expiry_map_barcode=products.set_index('barcode')['expiration_date'].to_dict()
products_catalog['expiration_date']=products_catalog['barcode'].map(expiry_map_barcode)
product_expiration_map = products_catalog.set_index('product_id')['expiration_date'].to_dict()
# Création d’un mapping barcode → price depuis products.csv

price_map=products.set_index('barcode')['price'].to_dict()
#Ajout des prix au catalogue produits
products_catalog['Price_DA']=products_catalog['barcode'].map(price_map) # mappe barcode → prix (NaN pour les 196 produits sans correspondance)
 
# Remplacement des valeurs manquantes par la médiane
products['price'] = pd.to_numeric(products['price'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
median_price=products['price'].median()
#remplace les NaN de prix par la médiane (270 DA)
products_catalog['Price_DA']=products_catalog['Price_DA'].fillna(median_price)

# Création d’un mapping barcode → allergens

allergens_map= products.set_index('barcode')['allergens'].to_dict()

products_catalog['allergens']=(products_catalog['barcode'].map(allergens_map)# ^ associe chaque barcode à ses allergènes
 
                               .fillna('none')) # ^ produits sans allergènes déclarés


products_catalog = products_catalog.rename(columns={
    'product_id'   : 'Product_ID',
    # ^ harmonise le nom de l'identifiant produit
    'product_name' : 'Name',
    # ^ harmonise le nom du produit
    'category'     : 'Category',
    # ^ harmonise le nom de la catégorie
})


# %%
# Conversion de la colonne 'timestamp' en format date (datetime)
# Cela permet de trier les données chronologiquement
interactions['timestamp'] = pd.to_datetime(interactions['timestamp'],errors='coerce')
interactions = interactions.dropna(subset=['timestamp'])

# Tri des interactions par ordre chronologique (du plus ancien au plus récent)
# reset_index(drop=True) permet de réinitialiser les index proprement
df_sorted  = interactions.sort_values('timestamp').reset_index(drop=True)

# Calcul de l'indice de séparation (80% des données pour train)
# len(df_sorted) donne le nombre total de lignes
split_idx  = int(len(df_sorted) * 0.8)

# Création du Train Set (données passées)
# On prend les 80% premières lignes (les plus anciennes)
train_df   = df_sorted.iloc[:split_idx].copy()

# Création du Test Set (données futures)
# On prend les 20% restantes (les plus récentes)
test_df    = df_sorted.iloc[split_idx:].copy()

# Affichage du nombre de lignes dans chaque dataset
print(f"Train: {len(train_df)} | Test: {len(test_df)}")
# Vérification : la fin du train doit être AVANT le début du test
print(f"Dernière date train : {train_df['timestamp'].max()}")
print(f"Première date test  : {test_df['timestamp'].min()}")

# %% [markdown]
# Elimination des Nan

# %%

print(f"\nNaN before cleaning : health_profile: {interactions['health_profile'].isna().sum()}")
#remplacer tous les NaN du  interactions,train et du test par standard
for df in [interactions, train_df, test_df]:
    df.columns = df.columns.str.strip()
for df in[interactions, train_df, test_df]:
    # 1. On remplace d'abord les 'None' textuels par 'Standard'
    df['health_profile']=df['health_profile'].replace('None', 'Standard').fillna('Standard')

     # ^ .str.split(',') : sépare la chaîne sur la virgule (allergens)
    df['health_profile_primary'] = (df['health_profile'].str.split('.').apply(lambda x : x[0].strip()))

# 3. Remplissage global sécurisé (évite le TypeError)
for df in [train_df, test_df]:
    # On remplit les colonnes numériques par 0
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # On remplit le reste (textes) par 'Standard' ou une chaîne vide
    obj_cols = df.select_dtypes(exclude='number').columns
    df[obj_cols] = df[obj_cols].fillna('Unknown')

print(f"Nettoyage terminé. NaN restants dans train : {train_df.isna().sum().sum()}")

# %% [markdown]
# Encodage -variable catégorielles

# %%
train_df['season']=train_df['season'].str.lower()
test_df['season']=test_df['season'].str.lower()

# ── Score implicite par action ── créé AVANT le One-Hot sur 'action'
action_weights = {
    'search'      : 1,   # recherche = signal d'intérêt très faible
    'view'        : 2,   # consultation = signal modéré
    'click'       : 2,   # clic = même niveau que view
    'add_to_cart' : 3,   # ajout au panier = intention d'achat forte
    'purchase'    : 4    # achat réel = signal maximum
}
#remplace chaque valeur de 'action' par son poids

train_df['action_score']=train_df['action'].map(action_weights)
test_df['action_score']=test_df['action'].map(action_weights)

#confirme que toutes les actions sont dans le dictionnaire
print(f"\naction_score NaN : {train_df['action_score'].isna().sum()}")
# vérifie la répartition des scores
print("Distribution :", train_df['action_score'].value_counts().sort_index().to_dict())

#______Label_Encoding->pour les variables avec un ordre natural

le=LabelEncoder()
#boucle sur 2 colonnes à encoder en Label Encoding

for col in['season','category']:
 # sécurité : passe à la colonne suivante si elle n'existe pas
   if col in train_df.columns:
        # On apprend les catégories sur l'UNION du train et du test
        # pour être sûr de ne rien rater
        full_categories = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(full_categories)
        # On transforme ensuite les deux
        train_df[col+'_enc'] = le.transform(train_df[col].astype(str))
        test_df[col+'_enc'] = le.transform(test_df[col].astype(str))
#_______One-hot encoding->pas d'ordre naturel entre ces modalités

#pour eviter la recréeation  de 'health_profile_primary' si le One-Hot l'a supprimée
user_health_profile = train_df[['user_id', 'health_profile_primary']].drop_duplicates('user_id').set_index('user_id')['health_profile_primary'].to_dict()


cols_ohe=[c for c in['health_profile_primary','action'] if c in train_df.columns]
if cols_ohe:
     prefixes = ['hp' if 'health' in c else 'act' for c in cols_ohe]
     train_df = pd.get_dummies(
        train_df,
        columns   = cols_ohe,
        # ^ colonnes à transformer en One-Hot
        prefix = prefixes,
        # ^ préfixe des nouvelles colonnes créées
        drop_first= True
        # ^ supprime la 1ère modalité → évite la multicolinéarité
     )
     test_df = pd.get_dummies(
        test_df,
        columns   = cols_ohe,
        prefix    = prefixes,
        drop_first= True
     )
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
print(f"One-Hot appliqué sur : {cols_ohe}")
print(f"Shape train : {train_df.shape} | test : {test_df.shape}")

# %% [markdown]
# ## Pahse2:ENRICHISSEMENT ET NORMALISATION

# %% [markdown]
# FEATURE SELECTION

# %%
cols_to_drop=['id', # identifiant technique → aucune valeur prédictive
              'timestamp',# utilisé uniquement pour split temporel
              'session_id', 
              'product_name',# texte redondant avec product_id
              'ingredients', # sera traité séparément avec TF-IDF
              'health_profile', # remplacé par health_profile_primary
              ]
# suppression des colonnes inutiles dans train et test
train_df.drop(
    columns= [c for c in cols_to_drop if c in train_df.columns], inplace=True)

test_df.drop(
    columns=[c for c in cols_to_drop if c in test_df.columns], inplace=True
)

# %% [markdown]
# FEATURE ENGINEERING

# %%
#  Ces features implémentent la logique métier du document :

#  score_context(i) = w_h × H_i + w_s × S_i + w_d × D_i
# ── H_i : compatibilité horaire ─────────────────────────────────
# 1. Calcul du score moyen par heure (TRAIN uniquement)
hour_scores = (
    train_df
    .groupby(['hour', 'category'])['action_score']
    .mean()
)
# ^ pour chaque heure (0-23) → action_score moyen des interactions
#   heure avec beaucoup d'achats → score élevé | peu d'achats → score bas

# 2. Normalisation MinMax → H_i ∈ [0, 1]
h_min = hour_scores.min()
h_max = hour_scores.max()
# ^ apprend min et max sur le TRAIN uniquement

hour_scores_norm = ((hour_scores - h_min) / (h_max - h_min)).to_dict()
# ^ formule MinMax : (x - min) / (max - min) → [0.0, 1.0]
#   .to_dict() : {heure → score_normalisé}

# 3. Application sur train et test
train_df['H_i'] = train_df.apply(
    lambda r: hour_scores_norm.get((r['hour'], r['category']), 0.5), axis=1
)
test_df['H_i'] = test_df.apply(
    lambda r: hour_scores_norm.get((r['hour'], r['category']), 0.5), axis=1
)



# ── S_i : compatibilité saisonnière par catégorie ───────────────
# Dataset : 2 saisons seulement d'apres dataset(winter et spring)

# Sécurité : season doit être en minuscule avant le groupby
train_df['season'] = train_df['season'].str.lower()
test_df['season']  = test_df['season'].str.lower()

#  1. Calcul de la popularité saison-catégorie (TRAIN uniquement)

season_popularity = (
    train_df
    .groupby(['season', 'category'])['action_score']
    .mean()
)
#2. Normalisation Min-Max 
min_val = season_popularity.min()
max_val = season_popularity.max()
#formule= (x - min) / (max - min)
season_popularity_norm = (
    (season_popularity - min_val) / (max_val - min_val)
)
season_popularity_norm = season_popularity_norm.to_dict()
# 3. Application sur train et test
train_df['S_i'] = train_df.apply(
    lambda row: season_popularity_norm.get(
        (row['season'], row['category']),0.5),axis=1)

test_df['S_i'] = test_df.apply(
    lambda row: season_popularity_norm.get(
        (row['season'], row['category']),0.5),axis=1)


# ── D_i : compatibilité DLC (anti-gaspillage) ───────────────────
from datetime import datetime

import pandas as pd

product_expiration_map = (
    products_catalog.set_index('Product_ID')['expiration_date']
    .to_dict()
)

def get_dynamic_d_i(expiration_date_raw):
 try:
    
     # ── 1. Reject None and bare Python NaN floats ──────────────
     if expiration_date_raw is None:
            return 0.5
     if isinstance(expiration_date_raw, float) and pd.isna(expiration_date_raw):
            return 0.5

        # ── 2. Already a Timestamp — use directly ──────────────────
     if isinstance(expiration_date_raw, pd.Timestamp):
            if pd.isnull(expiration_date_raw):   # catches pd.NaT
                return 0.5
            exp_date = expiration_date_raw

        # ── 3. String — parse without a rigid format ───────────────
     elif isinstance(expiration_date_raw, str):
            if expiration_date_raw.strip() in ('', 'None', 'nan', 'NaT'):
                return 0.5
            # No format= argument: lets pandas handle M/D/YYYY, ISO, etc.
            exp_date = pd.to_datetime(expiration_date_raw,
                                      dayfirst=False,
                                      errors='coerce')
            if pd.isnull(exp_date):
                return 0.5

        # ── 4. Anything else (datetime.datetime, numpy datetime64) ─
     else:
            exp_date = pd.Timestamp(expiration_date_raw)
            if pd.isnull(exp_date):
                return 0.5
     today = pd.Timestamp.now().normalize()
     days_left = (exp_date - today).days
     
     if days_left <= 0: return 0.0
     elif days_left <= 3:
        return 1.0 # URGENT : vendre immédiatement avant péremption
     elif days_left <= 7:
        return 0.8 # frais : haute priorité de recommandation
     elif days_left <= 30:
        return 0.5 # normal : priorité standard
     else:
        return 0.2 # longue durée : pas prioritaire
 except Exception as e:
  return 0.5

# ^ récupère dlc_days brut du train depuis df_sorted original
train_df['expiration_date'] = train_df['product_id'].map(product_expiration_map)
test_df['expiration_date'] = test_df['product_id'].map(product_expiration_map)

train_df['D_i']=train_df['expiration_date'].apply(get_dynamic_d_i) 
test_df['D_i']=test_df['expiration_date'].apply(get_dynamic_d_i) 

print(f"D_i créé — distribution : {train_df['D_i'].value_counts().to_dict()}")

#eviter le recalcule de dlc(une seul fois par jour)



# ── score_context : formule du document ─────────────────────────
w_h, w_s, w_d = 0.45, 0.30, 0.25
# vérifie que les poids somment à 1
assert abs(w_h + w_s + w_d - 1.0) < 1e-9

train_df['score_context']=(w_h * train_df['H_i']+ # heure
                           w_s* train_df['S_i']+ # heure
                           w_d * train_df['D_i']# DLC
                           )

test_df['score_context']=(w_h * test_df['H_i']+ # heure
                           w_s* test_df['S_i']+ # heure
                           w_d * test_df['D_i']# DLC
                           )
# ^ vérifie que le score moyen est dans [0,1] et raisonnable

print(f"\nscore_context — mean:{train_df['score_context'].mean():.3f}")




# %%
# ── score_session : formule du document ─────────────────────────
# Formule : score_session(i) = nb_interactions_produit_i / total_interactions_user

#nb_interactions_produit_i du train
sess_prod= train_df.groupby(['user_id','product_id'])['action_score'].transform('sum')

#total_interactions_user du train
sess_total=train_df.groupby(['user_id'])['action_score'].transform('sum')

#formule
train_df['score_session']=sess_prod/sess_total.replace(0,1)#.replace(0, 1) : évite la division par zéro si score_total = 0

#nb_interactions_produit_i du test
sp_t = test_df.groupby(['user_id', 'product_id'])['action_score'].transform('sum')
#total_interactions_user du test
st_t = test_df.groupby('user_id')['action_score'].transform('sum')
test_df['score_session'] = sp_t / st_t.replace(0, 1)
print(f"score_session — mean:{train_df['score_session'].mean():.3f}")

# ── Popularité produit ───────────────────────────────────────────

popularity = (
    train_df.groupby('product_id')['action_score']
    .sum()
    # ^ somme de tous les action_scores par produit dans le train
    .rename('popularity')
    # ^ renomme la Series résultante
)
# calculée sur TRAIN uniquement → évite que les achats futurs influencent le score
 
train_df = train_df.merge(popularity, on='product_id', how='left')
# ^ .merge() : joint la popularité dans train_df
#   on='product_id' : clé de jointure
#   how='left' : garde toutes les lignes du train (même sans popularité)
 
test_df = test_df.merge(popularity, on='product_id', how='left')
# ^ même jointure sur le test
 
test_df['popularity'] = test_df['popularity'].fillna(0)
# ^ produits du test jamais vus dans le train → popularité = 0
 
print(f"popularité — max:{train_df['popularity'].max():.0f}")

# %% [markdown]
#  FEATURE SCALING
# 

# %%
#scaling de Price_DA
price_map_full=products_catalog.set_index('Product_ID')['Price_DA'].to_dict()

train_df['Price_DA']=train_df['product_id'].map(price_map_full).fillna(median_price)
test_df['Price_DA']=test_df['product_id'].map(price_map_full).fillna(median_price)

# MinMaxScaler → Price_DA

# ^ instancie le scaler — pas encore ajusté
mms=MinMaxScaler()
train_df['price_scaled']=mms.fit_transform(train_df[['Price_DA']])

test_df['price_scaled']=mms.transform(test_df[['Price_DA']])

#scaling de dlc_days
# StandardScaler → dlc_days
dlc_train_2d = df_sorted.iloc[:split_idx][['dlc_days']].values
dlc_test_2d  = df_sorted.iloc[split_idx:][['dlc_days']].values

ss = StandardScaler()
train_df['dlc_scaled'] = ss.fit_transform(dlc_train_2d)
test_df['dlc_scaled'] = ss.transform(dlc_test_2d)

# StandardScaler → action_score (valeurs 1,2,3,4)
ss2 = StandardScaler()
train_df['score_scaled'] = ss2.fit_transform(train_df[['action_score']])
test_df['score_scaled']  = ss2.transform(test_df[['action_score']])

# Suppression des colonnes brutes remplacées par leurs versions scalées

for col in ['Price_DA']:
    if col in train_df.columns: train_df.drop(columns=[col], inplace=True)
    if col in test_df.columns:  test_df.drop(columns=[col], inplace=True)

print(f"\nScaling appliqué :")
print(f"  price_scaled  → [{train_df['price_scaled'].min():.2f}, "
      f"{train_df['price_scaled'].max():.2f}]")
 
print(f"  dlc_scaled    → µ≈{train_df['dlc_scaled'].mean():.2f}, "
      f"σ≈{train_df['dlc_scaled'].std():.2f}")
 
print(f"  score_scaled  → µ≈{train_df['score_scaled'].mean():.2f}, "
      f"σ≈{train_df['score_scaled'].std():.2f}")


# %% [markdown]
#   SUPPRESSION DES OUTLIERS NÉFASTES
# 

# %%
#compte les interactions par user dans le train
user_counts=train_df.groupby('user_id').size()
print(f"\ninteractions/user -min:{user_counts.min()},"
      f"médiane:{user_counts.median():.0f},max:{user_counts.max()}")

print(f"user<3 interactions:{(user_counts<3).sum()}")
# ^ .quantile([0.25, 0.75]) : calcule Q1 et Q3 (quartiles 1 et 3)
products_catalog['Price_DA'] = pd.to_numeric(products_catalog['Price_DA'], errors='coerce')
q1_p, q3_p = products_catalog['Price_DA'].quantile([0.25, 0.75])
# ^ IQR = Inter-Quartile Range = Q3 - Q1 = amplitude centrale
iqr = q3_p - q1_p
# ^ règle de Tukey : outlier si x > Q3 + 1.5 × IQR

outliers_prix = products_catalog[products_catalog['Price_DA'] > q3_p + 1.5 * iqr]
print(f"\nPrix outliers (> {q3_p + 1.5*iqr:.0f} DA) : "
      f"{len(outliers_prix)} produits CONSERVÉS")
print(outliers_prix[['Name', 'Price_DA']].sort_values('Price_DA', ascending=False)
      .head(8).to_string(index=False))

# %% [markdown]
# #  PHASE 3 — FEATURE ENGINEERING CONTENT-BASED

# %%
# Nettoyage du texte des ingrédients
products_catalog['ingredients_clean']=(products_catalog['ingredients'].fillna('').str.replace(r'\b(fresh|no|preservatives|added|natural|vary|product|by)\b', '', regex=True)
.str.strip())
# 2. Configuration du Vectorizer
custom_stopwords = [
    'ingredients', 'contains', 'water', 'and', 'with',
    'products', 'daily', 'catch', 'added'  ,'100'
]
#texte normalisé prêt pour TF-IDF
tfidf=TfidfVectorizer(stop_words=custom_stopwords,max_features=100,ngram_range=(1,2),min_df=2,sublinear_tf = True     # ^ utilise log(1 + TF) au lieu de TF → atténue les termes sur-représentés

)
# cree matrice sparse
tfidf_matrix=tfidf.fit_transform(products_catalog['ingredients_clean'])
print(f"\nMatrice TF-IDF : {tfidf_matrix.shape}")
print(f"top 10 termes: {tfidf.get_feature_names_out()[:10].tolist()}")

#  FILTRAGE PRODUITS DU TRAIN (ANTI DATA LEAKAGE)
# Liste des produits présents dans le train uniquement
train_pids=train_df['product_id'].unique()

# On garde juste produits du train pour éviter fuite d'information
products_train=(products_catalog[products_catalog['Product_ID'].isin(train_pids)].copy().reset_index(drop=True))

tfidf_train = tfidf.transform(products_train['ingredients_clean'])

# %% [markdown]
#  COSINE SIMILARITY (CALCUL DES SIMILARITES)
# 

# %%
from sklearn.metrics.pairwise import cosine_similarity
# Compare chaque produit avec tous les autres
sim_matrix = cosine_similarity(tfidf_train, tfidf_train)

sim_df = pd.DataFrame(
    sim_matrix,
    index=products_train['Product_ID'],   # lignes = produits
    columns=products_train['Product_ID']  # colonnes = produits
)

print(f"Similarity matrix shape : {sim_df.shape}")


# %% [markdown]
# FONCTION DE RECOMMANDATION (PRODUITS SIMILAIRES)

# %%
def get_similar_products(product_id,n=5):
    # product_id : ID du produit de référence
    #  n          : nombre de recommandations
    if product_id not in sim_df.index:
         # Produit inconnu → retourne vide
         return pd.Series(dtype=float)
    scores =sim_df[product_id].drop(product_id) # exclut le produit lui-même (sim=1.0 avec soi-même)
    return scores.nlargest(n)
#test de la fct
sample_pid = train_pids[0]
sample_name = products_catalog.loc[
    products_catalog['Product_ID'] == sample_pid, 'Name'
].values[0]  # values[0] car ça renvoie un array
print(f"\nTest get_similar_products of {sample_name} ,its id: {sample_pid} :")
sims = get_similar_products(sample_pid)
for pid, score in sims.items():
    name = products_catalog[products_catalog['Product_ID'] == pid]['Name'].values
    name = name[0][:25] if len(name) > 0 else '?'
    print(f"  {pid} ({name}) → {score:.3f}")

# %% [markdown]
# FILTRAGE SANTÉ

# %%
import re                       # Expressions régulières pour nettoyage de texte
# ── 1. Dictionnaire de normalisation linguistique ─────────
NORMALIZATION_MAP = {
    'lait':'milk', 'lait entier':'milk', 'lactos':'lactose', 'beurre':'butter',
    'crème':'cream', 'fromage':'cheese', 'yaourt':'milk', 'yogurt':'milk',
    'farine':'flour', 'blé':'wheat', 'orge':'barley', 'seigle':'rye',
    'sucre':'sugar', 'glucose':'sugar', 'fructose':'sugar', 'sirop':'syrup', 'miel':'sugar',
    'saccharose':'sugar', 'poulet':'chicken', 'bœuf':'beef', 'agneau':'meat', 'porc':'meat',
    'dinde':'chicken', 'crevette':'shrimp', 'thon':'tuna', 'saumon':'fish', 'cabillaud':'fish',
    'calamar':'seafood', 'sel':'salt', 'sels':'salt', 'noisette':'hazelnut', 'noix':'nuts',
    'amande':'almonds', 'cacahuète':'peanuts', 'arachide':'peanuts'
}

# ── 2. Synonymes médicaux ──────────────────────────────────
MEDICAL_SYNONYMS = {
    'casein':'milk', 'caseinate':'milk', 'whey':'milk', 'lactalbumin':'milk', 'lactoglobulin':'milk',
    'ghee':'butter', 'fromage blanc':'cheese', 'ricotta':'cheese', 'mascarpone':'cheese',
    'semolina':'wheat', 'semoule':'wheat', 'spelt':'wheat', 'kamut':'wheat', 'triticale':'wheat',
    'malt':'barley', 'maltodextrin':'barley', 'dextrose':'sugar', 'maltose':'sugar', 'sucrose':'sugar',
    'sorbitol':'sugar', 'corn syrup':'syrup', 'agave':'syrup', 'lard':'fat', 'tallow':'fat',
    'margarine':'butter', 'shortening':'fat', 'monosodium':'sodium', 'bicarbonate':'sodium',
    'baking soda':'sodium', 'soy sauce':'salt', 'praline':'hazelnut', 'marzipan':'almonds',
    'tahini':'nuts', 'nougat':'hazelnut', 'anchovy':'fish', 'anchois':'fish', 'surimi':'fish',
    'cod':'fish', 'haddock':'fish', 'prawn':'shrimp', 'langoustine':'shrimp',
    'crab':'seafood', 'lobster':'seafood', 'oyster':'seafood', 'mussel':'seafood', 'scallop':'seafood'
}
# ── 3. Fusion des dictionnaires ────────────────────────────
FULL_NORMALIZATION = {**NORMALIZATION_MAP, **MEDICAL_SYNONYMS}

# ── 4. Profils santé avec mots-clés normalisés ─────────────
HEALTH_KEYWORDS = {
    'Diabetes': {'sugar', 'soda', 'syrup', 'nectar', 'juice', 'candy', 
        'sweetened', 'jam', 'chocolate', 'biscuit', 'energy drink'},
    'Gluten intolerance': {'wheat', 'flour', 'pasta', 'couscous', 'semolina', 
        'bread', 'pastry', 'spaghetti', 'macaroni'},
    'Celiac disease': {'gluten', 'wheat', 'flour', 'barley', 'rye', 'pasta', 'couscous', 'semolina', 'spaghetti', 'macaroni', 'biscuit'},
    'Lactose intolerance': {'milk', 'cheese', 'yogurt', 'butter', 'lactic', 'dairy', 
        'emmental', 'camembert', 'lben', 'cream', 'petit suisse'},
    'Nut allergy': {'nuts', 'peanuts', 'almonds', 'hazelnut', 'pistachio', 'walnut', 'cashew'},
    'Seafood allergy': {'fish', 'shrimp', 'seafood', 'tuna', 'squid', 'mackerel', 'sardines', 'whiting', 'sea bream', 'anchovies', 'catch'},
    'High blood pressure': {'salt', 'salted', 'sodium', 'pickles', 'canned', 
        'processed cheese', 'chips', 'anchovies'},
    'High cholesterol': {'fat','butter','cream','beef','lamb','merhues','sausage','tenderloi','mutton','chops'},
    'Keto diet': {'sugar','flour','pasta','rice'},
    'Vegetarian': {'meat','chicken','beef','fish'},
    'Vegan': {'meat', 'chicken', 'beef', 'lamb', 'turkey', 'fish', 
        'shrimp', 'squid', 'mackerel', 'sardines', 'tuna', 'whiting', 
        'sea bream', 'milk', 'cheese', 'yogurt', 'butter', 'egg', 
        'honey', 'merguez', 'sausage', 'meatballs', 'lben'},
    'Obesity': {'sugar', 'syrup', 'chocolate', 'soda', 'candy', 'jam', 'sweetened', 'energy drink', 'juice'},
    'Standard': set()
}

# ── 5. Normalisation des textes ────────────────────────────
def normalize_text(text: str) -> str:
    if pd.isna(text):                # Vérifie si le texte est NaN
        return ""
    text = str(text).lower()         # Minuscule
    text = re.sub(r'[^a-zàâäéèêëîïôùûüç0-9 ]', ' ', text)  # Nettoyage ponctuation
    tokens = text.split()            # Découpage en mots
    normalized = [FULL_NORMALIZATION.get(tok, tok) for tok in tokens]  # Normalisation
    return ' '.join(normalized)      # Retour texte normalisé

# ── 6. Concaténation texte produit ─────────────────────────
def build_product_text(row) -> str:
    parts = [str(row.get('ingredients','')), str(row.get('allergens','')), str(row.get('Name',''))]
    return normalize_text(' '.join(parts))  # Normalisation complète

# ── 7. Préparation du catalogue ────────────────────────────
products_catalog['normalized_text'] = products_catalog.apply(build_product_text, axis=1)
products_catalog['token_set'] = products_catalog['normalized_text'].apply(lambda x: set(x.split()))

# ── 8. TF-IDF pour couche 3 ────────────────────────────────
tfidf_layer3 = TfidfVectorizer(max_features=200, ngram_range=(1,2), min_df=2, sublinear_tf=True)
tfidf_matrix_l3 = tfidf_layer3.fit_transform(products_catalog['normalized_text'])
tfidf_dense = tfidf_matrix_l3.toarray()  # Matrice dense pour rapidité

# ── 9. Vecteurs profils santé TF-IDF ──────────────────────
health_profile_vectors = {
    profile: tfidf_layer3.transform([' '.join(keywords)])
    for profile, keywords in HEALTH_KEYWORDS.items() if keywords
}
# ── 10. Détection produit interdit ────────────────────────
def is_forbidden_product(product_idx:int, token_set:set, forbidden_keywords:set, profile_name:str=None, tfidf_threshold:float=0.25) -> bool:
    if token_set & forbidden_keywords:    # Couche 1 : intersection
        return True
    if profile_name and profile_name in health_profile_vectors:  # Couche 2
        profile_vec = health_profile_vectors[profile_name]
        product_vec = tfidf_dense[product_idx].reshape(1,-1)
        sim = cosine_similarity(profile_vec, product_vec)[0][0]
        if sim > tfidf_threshold:
            return True
    return False

# ── 11. Pré-filtrage  ──────────────────────────────
def prefiltrage_sante(candidates_df: pd.DataFrame, health_profile:str, use_tfidf:bool=True) -> pd.DataFrame:
    primary = health_profile.split(',')[0].strip()
    if primary not in HEALTH_KEYWORDS:
        print(f" Profil '{primary}' inconnu")
        return candidates_df
    forbidden = HEALTH_KEYWORDS[primary]
    if not forbidden: return candidates_df

    results = []
    for idx, row in candidates_df.iterrows():
        product_idx = products_catalog.index.get_loc(idx) if idx in products_catalog.index else candidates_df.index.get_loc(idx)
        forbidden_flag = is_forbidden_product(
            product_idx=product_idx,
            token_set=row['token_set'],
            forbidden_keywords=forbidden,
            profile_name=primary if use_tfidf else None,
            tfidf_threshold=0.25
        )
        results.append(not forbidden_flag)  # True = produit sûr
    filtered = candidates_df[results].copy()
    n_excluded = len(candidates_df)-len(filtered)
    print(f"[{primary}] couche {'1+2' if use_tfidf else '1+2'} : {n_excluded} exclus → {len(filtered)} restants")
    return filtered

# ── 12. Post-filtrage  ───────────────────────────────
def postfiltrage_sante(recommended_ids:list, health_profile:str) -> list:
    primary = health_profile.split(',')[0].strip()
    forbidden = HEALTH_KEYWORDS.get(primary,set())
    if not forbidden: return recommended_ids
    id_to_tokens = products_catalog.set_index('Product_ID')['token_set'].to_dict()
    return [pid for pid in recommended_ids if not is_forbidden_product(products_catalog.index.get_loc(products_catalog[products_catalog['Product_ID']==pid].index[0]), id_to_tokens.get(pid,set()), forbidden)]

# ── 13. Ajouter profil santé dynamiquement ───────────────
def add_health_profile(profile_name:str, raw_keywords:list, normalize:bool=True) -> None:
        if normalize:
                normalized_kw = {FULL_NORMALIZATION.get(k.lower(), k.lower()) for k in raw_keywords}
        else:
                            normalized_kw = {k.lower() for k in raw_keywords}
        HEALTH_KEYWORDS[profile_name] = normalized_kw


# %% [markdown]
# TESTS DU FILTRAGE SANTÉ

# %%
# Test 1 : Pré-filtrage
filtered = prefiltrage_sante(products_catalog.copy(), 'Vegan')
filtered_out = products_catalog[
    ~products_catalog.index.isin(filtered.index)
]

print("Produits exclus ")
print(filtered_out[['Name', 'ingredients']].head(10))

print()

# %%
#Test 4 : ajout profil
add_health_profile('Low sodium diet', ['sel', 'sodium', 'salt', 'sels'])
filtered_sodium = prefiltrage_sante(products_catalog.copy(), 'Low sodium diet')

# %%
#Test 5 : Post-filtrage
sample_ids = products_catalog['Product_ID'].sample(20).tolist()
safe_ids = postfiltrage_sante(sample_ids, 'Lactose intolerance')

print("Total:", len(sample_ids))
print("Safe :", len(safe_ids))
removed_ids=set(sample_ids) - set(safe_ids)
print("Removed:", removed_ids)

removed_products = products_catalog[
    products_catalog['Product_ID'].isin(removed_ids)
]

print(removed_products[['Product_ID','Name', 'ingredients']])


# %% [markdown]
# Score final pondéré 

# %%
def calculer_score_final(user_type, score_context_val,
                          score_session_val=0.0, score_contenu_val=0.0):
    if user_type == 2:
        score = 0.5 * score_session_val + 0.5 * score_context_val
        #  w1=0.5, w2=0.5 : importance égale session et contexte
        #   utilisé quand le score_contenu n'est pas disponible (pas d'historique)
    elif user_type == 1:
        score= score_context_val
    else:  # Type 3
         #  w1=0.6, w2=0.4 
        score = 0.6 * score_contenu_val + 0.4 * score_context_val
       
 
    return round(float(np.clip(score, 0, 1)), 4)
    #  np.clip(score, 0, 1) : garantit que le score ∈ [0.0, 1.0]
    #   float() : convertit en float Python standard
    #   round(..., 4) : arrondit à 4 décimales (lisibilité)
 
train_df['score_final'] = train_df.apply(
    lambda row: calculer_score_final(
     # simplifié : type 3 pour toutes les interactions
        user_type         = 3,
        score_context_val = row['score_context'],
        score_session_val = row['score_session'],
        score_contenu_val = 0.5
        # ^ placeholder → remplacer par la vraie similarité cosinus (sim_df)
    ),
    axis=1
    # axis=1 : applique la fonction sur chaque ligne 
)
 
print("\nScore final — aperçu :")
print(train_df[['user_id', 'product_id', 'score_context',
                 'score_session', 'score_final']].head(8).to_string())
    

# %% [markdown]
# Métriques d'évaluation propres à la recommandation

# %%
# ── Métriques d'évaluation pour le système de recommandation ──
# RMSE seul ne suffit pas → on utilise Precision@K, Recall@K, NDCG@K

def precision_at_k(recommended, relevant, k=5):
    """Proportion des K premières recommandations pertinentes"""
    if not recommended or not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k=5):
    """Proportion des produits pertinents retrouvés dans le top-K"""
    if not recommended or not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)

def ndcg_at_k(recommended, relevant, k=5):
    """Score pondéré selon la position des bonnes recommandations"""
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(top_k) if p in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

# Exemple simple pour montrer le calcul
rec_example = [234, 221, 233, 246, 201]  # produits recommandés
rel_example = {234, 221}                 # produits réellement pertinents

print("\nTest métriques (exemple Yogurts) :")
print(f"  Recommandés : {rec_example}")
print(f"  Pertinents  : {rel_example}")
print(f"  Precision@5 : {precision_at_k(rec_example, rel_example, 5):.2f}")
print(f"  Recall@5    : {recall_at_k(rec_example, rel_example, 5):.2f}")
print(f"  NDCG@5      : {ndcg_at_k(rec_example, rel_example, 5):.2f}")

# %%
import pickle, joblib
from pathlib import Path

PKL_OUT = Path().resolve().parent / "fastapi-supermarche"/ "models_pkl"
PKL_OUT.mkdir(exist_ok=True)

joblib.dump(tfidf, PKL_OUT / "tfidf.pkl")
pickle.dump(sim_df, open(PKL_OUT / "sim_df.pkl", "wb"))
pickle.dump(products_catalog, open(PKL_OUT / "products_catalog.pkl", "wb"))
pickle.dump(train_df, open(PKL_OUT / "train_df.pkl", "wb"))
pickle.dump(test_df, open(PKL_OUT / "test_df.pkl", "wb"))

pickle.dump(hour_scores_norm, open(PKL_OUT / "hour_scores_norm.pkl", "wb"))
pickle.dump(season_popularity_norm, open(PKL_OUT / "season_popularity_norm.pkl", "wb"))
pickle.dump(user_health_profile, open(PKL_OUT / "user_health_profile.pkl", "wb"))
pickle.dump(product_expiration_map, open(PKL_OUT / "product_expiration_map.pkl", "wb"))

print(f" Tous les .pkl sauvegardés dans : {PKL_OUT}")


