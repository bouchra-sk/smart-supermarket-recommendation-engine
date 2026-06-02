from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from uuid import UUID
from database import get_async_session
from models.user_model import User
from services.recommendation_service import RecommendationService
from auth_service import current_active_user

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

@router.get("/me")
async def get_my_recommendations(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """
    Get personalized recommendations for the current user.
    Uses ML model from modeling.py via RecommendationService.
    """
    
    recommendations = await RecommendationService.get_user_recommendations(
        db, current_user.id, limit
    )
    
    return {
        "user_id": str(current_user.id),
        "recommendations": recommendations,
        "count": len(recommendations),
        "source": "recommendation_engine"
    }

@router.post("/batch")
async def get_batch_recommendations(
    user_ids: List[UUID],
    db: AsyncSession = Depends(get_async_session)
):
    """Get recommendations for multiple users (for admin/batch processing)"""
    
    results = {}
    for user_id in user_ids:
        recommendations = await RecommendationService.get_user_recommendations(db, user_id)
        results[str(user_id)] = recommendations
    
    return {
        "users": results,
        "source": "recommendation_engine"
    }