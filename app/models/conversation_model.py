from sqlalchemy import Column, Integer, ForeignKey
from app.db.database import Base


class Conversation(Base):

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
