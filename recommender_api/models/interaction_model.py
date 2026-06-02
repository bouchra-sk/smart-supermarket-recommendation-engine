from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()
#from database import Base
class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    session_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    health_profile = Column(String, nullable=True)
    
    product_id = Column(Integer, nullable=False)
    product_name = Column(String, nullable=False)
    barcode = Column(String, nullable=True)
    category = Column(String, nullable=True)
    ingredients = Column(String, nullable=True)
    season = Column(String, nullable=True)

    action = Column(String, nullable=False)  # view, click, purchase, etc.
    dlc_days = Column(Integer, nullable=True)
    hour = Column(Integer, nullable=True)