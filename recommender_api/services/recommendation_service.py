import os
import sys
import pickle
import logging
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from models.product_model import Product, Category
from models.interaction_model import Interaction
from models.user_model import User

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

 #─── Scores d'action (poids relatifs) ────────────────────────────────────────
# Ces valeurs doivent correspondre à celles utilisées pendant l'entraînement
ACTION_SCORES = {
    "purchase": 3,
    "add_to_cart": 2,
    "view": 1,
}
# Fenêtre de temps pour les interactions "récentes" (score session)
SESSION_WINDOW_HOURS = 24

class RecommendationService:
    
    @staticmethod
    def get_season_now():
        month = datetime.now().month
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        return 'autumn'
    @staticmethod
    async def _fetch_user_history(db: AsyncSession, user_id) -> list[int]:
        
        result = await db.execute(
            select(Interaction.product_id)
            .where(Interaction.user_id == user_id)
            .where(Interaction.action == "purchase")
            .distinct()
        )
        return [row[0] for row in result.all()]
 
    @staticmethod
    async def _fetch_session_data(db: AsyncSession, user_id) -> pd.DataFrame:
        
        since = datetime.utcnow() - timedelta(hours=SESSION_WINDOW_HOURS)
 
        result = await db.execute(
            select(
                Interaction.product_id,
                Interaction.action,
            )
            .where(Interaction.user_id == user_id)
            .where(Interaction.timestamp >= since)
            .order_by(Interaction.timestamp.desc())
        )
        rows = result.all()
 
        if not rows:
            return pd.DataFrame()
 
        records = []
        for product_id, action in rows:
            records.append({
                "user_id":      str(user_id),
                "product_id":   product_id,
                "action":       action,
                "action_score": ACTION_SCORES.get(action, 1),
            })
 
        df = pd.DataFrame(records)
 
        # Si un même produit apparaît plusieurs fois, on somme les scores
        # (cohérent avec le groupby dans modeling.py ligne 249)
        df = (
            df.groupby(["user_id", "product_id"], as_index=False)
            .agg(action_score=("action_score", "sum"))
        )
        return df
    
    @staticmethod
    async def get_user_recommendations(db: AsyncSession, user_id, limit=10):
        
        print(f"\n{'='*50}")
        print(f" Getting recommendations for user: {user_id}")
        print(f"{'='*50}")
        
        try:
            # 1. Recuperation des donnees SQL
            purchased_ids = await RecommendationService._fetch_user_history(db, user_id)
            session_df    = await RecommendationService._fetch_session_data(db, user_id)

            # Profil sante de l'utilisateur
            user_result    = await db.execute(select(User.health_profile).where(User.id == user_id))
            health_profile = user_result.scalar_one_or_none() or "Standard"

            print(f"  -> {len(purchased_ids)} produit(s) achetes en base")
            print(f"  -> {len(session_df)} interaction(s) recentes")
            print(f"  -> Profil sante : {health_profile}")

            # 2. Import du moteur ML
            print("Importing modeling.py...")
            from src_sys.modeling import recommander
            print("recommander imported")

            # 3. Contexte temporel
            now            = datetime.now()
            current_hour   = now.hour
            current_season = RecommendationService.get_season_now()

            print(f"Calling recommander with Hour={current_hour}, Season={current_season}...")

            recommendations_df = recommander(
                user_id=str(user_id),
                n=limit,
                current_hour=current_hour,
                current_season=current_season,
                force_user_type=None,
                session_data=session_df if not session_df.empty else None,
                sql_history=purchased_ids if purchased_ids else None,
            )

            if recommendations_df is not None and not recommendations_df.empty:
                print(f" ML model returned {len(recommendations_df)} recommendations")

                # Construire le reason dynamiquement
                reason_parts = [f"Recommended for {current_season}"]
                if health_profile and health_profile.lower() != "standard":
                    reason_parts.append(f"adapted to your {health_profile} diet")
                if purchased_ids:
                    reason_parts.append(f"based on your {len(purchased_ids)} past purchase(s)")
                reason = " and ".join(reason_parts)

                return [
                    {
                        "id":      int(row["product_id"]),
                        "name":    row["Name"],
                        "price":   float(row.get("Price_DA") or row.get("price") or 0),
                        "barcode": str(row.get("barcode") or ""),
                        "score":   float(row.get("score_final", 0)),
                        "reason":  reason,
                    }
                    for _, row in recommendations_df.iterrows()
                ]
            else:
                print(" ML returned empty DataFrame")
                
        except Exception as e:
            print(f"ML recommendation error: {e}")
            import traceback
            traceback.print_exc()
                # ── Fallback ──────────────────────────────────────────────────────

        print(" Using fallback recommendations (Category-based)")
        return await RecommendationService._get_fallback_recommendations(db, user_id, limit)
     # Fallback inchangé

    @staticmethod
    async def _get_fallback_recommendations(db: AsyncSession, user_id, limit=10):
        
        result = await db.execute(
            select(Interaction.category, func.count().label('count'))
            .where(Interaction.user_id == user_id)
            .where(Interaction.category.isnot(None))
            .group_by(Interaction.category)
            .order_by(func.count().desc())
            .limit(3)
        )
        category_names = result.all()
        
        if not category_names:
            result = await db.execute(
                select(Product).where(Product.is_available == True).limit(limit)
            )
            products = result.scalars().all()
            return [{"id": p.id, "name": p.name, "price": p.price, "reason": "Popular products"} for p in products]
        
        recommendations = []
        for cat_name, _ in category_names:
            product_result = await db.execute(
                select(Product).join(Category).where(Category.name == cat_name).limit(3)
            )
            recommendations.extend(product_result.scalars().all())

        return [
            {
                "id": p.id,
                "name": p.name,
                "price": p.price,
                "reason": "Based on your interests"
            }
            for p in recommendations[:limit]
        ]