from sqlalchemy import Column, Integer, Text, ForeignKey
from app.db.database import Base


class ChatHistory(Base):

    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id")
    )

    user_message = Column(Text)

    ai_response = Column(Text)
