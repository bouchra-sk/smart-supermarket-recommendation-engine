# routers/interaction_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from database import get_async_session
from models.interaction_model import Interaction
from schemas.interaction_schema import InteractionCreate, InteractionRead

router = APIRouter(prefix="/interactions", tags=["interactions"])

# Create interaction
@router.post("/", response_model=InteractionRead)
async def create_interaction(interaction: InteractionCreate, db: AsyncSession = Depends(get_async_session)):
    db_interaction = Interaction(**interaction.dict())
    db.add(db_interaction)
    await db.commit()
    await db.refresh(db_interaction)
    return db_interaction

# Get interactions by user
@router.get("/user/{user_id}", response_model=List[InteractionRead])
async def get_user_interactions(user_id: int, db: AsyncSession = Depends(get_async_session)):
    result = await db.execute(
        select(Interaction).where(Interaction.user_id == user_id)
    )
    interactions = result.scalars().all()
    return interactions

# Get interactions by product
@router.get("/product/{product_id}", response_model=List[InteractionRead])
async def get_product_interactions(product_id: int, db: AsyncSession = Depends(get_async_session)):
    result = await db.execute(
        select(Interaction).where(Interaction.product_id == product_id)
    )
    interactions = result.scalars().all()
    return interactions