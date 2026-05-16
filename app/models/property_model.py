from sqlalchemy import Column, Integer, String
from app.db.database import Base


class Property(Base):

    __tablename__ = 'properties'

    id = Column(Integer, primary_key=True, index=True)

    title = Column(String)

    location = Column(String)

    price = Column(Integer)

    bhk = Column(Integer)

    description = Column(String)
