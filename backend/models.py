from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, default="demo_user")
    description = Column(String)
    arm = Column(Integer, nullable=True)
    base_price = Column(Float, nullable=True)
    quoted_price = Column(Float, nullable=True)
    outcome = Column(String, default="pending")   
    final_price = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)