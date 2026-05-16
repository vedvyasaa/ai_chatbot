from app.db.database import engine, Base

from app.models.user_model import User
from app.models.conversation_model import Conversation
from app.models.chat_model import ChatHistory
from app.models.property_model import Property

Base.metadata.create_all(bind=engine)

print("Tables created successfully.")
