from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class InteractionBase(BaseModel):
    user_id: int
    session_id: Optional[str]
    timestamp: Optional[datetime]
    health_profile: Optional[str]

    product_id: int
    product_name: str
    barcode: Optional[str]
    category: Optional[str]
    ingredients: Optional[str]
    season: Optional[str]

    action: str
    dlc_days: Optional[int]
    hour: Optional[int]

class InteractionCreate(InteractionBase):
    pass

class InteractionRead(InteractionBase):
    id: int

    class Config:
        orm_mode = True