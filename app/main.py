from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import UploadFile, File
import shutil
from app.services.rag_service import (
    extract_pdf_text,
    chunk_text,
    create_chunk_embeddings,
    build_chunk_index,
    search_similar_chunks
)
from app.services.ai_service import ask_ai
from app.services.property_service import search_properties
from app.services.nlp_service import extract_preferences
from app.services.vector_service import (
    create_embeddings, build_faiss_index, semantic_search)
from app.services.chroma_service import (
    store_document_chunks,
    search_document
)
from app.services.rag_service import model
from app.services.agent_service import (
    detect_intent,
    decide_tools
)
from app.db.database import SessionLocal
from app.models.chat_model import ChatHistory
from app.models.user_model import User
from app.models.conversation_model import Conversation
from app.models.property_model import Property

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    conversation_id: int
    message: str


class UserRequest(BaseModel):
    name: str
    email: str
    phone: str


class ConversationRequest(BaseModel):
    user_id: int


class PropertyRequest(BaseModel):
    title: str
    location: str
    price: int
    bhk: int
    description: str


@app.get("/")
def home():
    return {"message": "AI Chatbot Backend Running"}


@app.post("/chat")
def chat(request: ChatRequest):

    db = SessionLocal()

    try:

        previous_chats = db.query(ChatHistory).filter(
            ChatHistory.conversation_id == request.conversation_id
        ).order_by(
            ChatHistory.id.desc()
        ).limit(5).all()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful real estate AI assistant."
            }
        ]

        for chat in previous_chats:

            messages.append({
                "role": "user",
                "content": chat.user_message
            })

            messages.append({
                "role": "assistant",
                "content": chat.ai_response
            })

        messages.append({
            "role": "user",
            "content": request.message
        })

        intent = detect_intent(
            request.message
        )

        tools = decide_tools(intent)

        print("Intent:", intent)
        print("Tools:", tools)

        # Search filters
        # location = None
        # bhk = None

        # message_lower = request.message.lower()

        # if "mumbai" in message_lower:
        #     location = "Mumbai"

        # if "2bhk" in message_lower or "2 bhk" in message_lower:
        #     bhk = 2

        if "postgres_search" in tools:

            preferences = extract_preferences(
                request.message
            )

            if isinstance(preferences, dict):

                location = preferences.get("location")
                bhk = preferences.get("bhk")
                budget = preferences.get("budget")

            else:

                location = None
                bhk = None
                budget = None

            properties = search_properties(
                db,
                location=location,
                bhk=bhk,
                max_price=budget
            )

        all_properties = db.query(Property).all()

        semantic_results = []

        if all_properties:

            embeddings = create_embeddings(all_properties)

            index = build_faiss_index(embeddings)

            semantic_results = semantic_search(
                request.message,
                all_properties,
                index
            )

        if semantic_results:

            property_text = "\n".join([
                f"""
                Title: {p.title}
                Location: {p.location}
                Price: ₹{p.price}
                Description: {p.description[:100]}
                """
                for p in semantic_results
            ])

            messages.append({
                "role": "system",
                "content": f"""
Properties:
{property_text}
"""
            })

        search_results = None

        if "chroma_rag_search" in tools:

            query_embedding = model.encode(
                [request.message]
            )

            # search_document will now return dict with lists even if empty
            search_results = search_document(
                query_embedding
            )

        # Ensure search_results has lists and is non-empty before using
        if search_results and isinstance(search_results, dict) and len(search_results.get("documents", [])) > 0:
            relevant_chunks = search_results.get("documents", [])
            metadata_results = search_results.get("metadatas", [])

            chunk_entries = []
            for i in range(min(2, len(relevant_chunks))):
                meta_source = metadata_results[i].get('source') if i < len(
                    metadata_results) and isinstance(metadata_results[i], dict) else 'unknown'
                chunk_entries.append(f"""
Source: {meta_source}

Content:
{relevant_chunks[i]}
""")

            chunk_text_content = "\n".join(chunk_entries)

            messages.append({
                "role": "system",
                "content": f"""
Document Context:

{chunk_text_content}
"""
            })

        ai_response = ask_ai(messages)

        new_chat = ChatHistory(
            conversation_id=request.conversation_id,
            user_message=request.message,
            ai_response=ai_response
        )

        db.add(new_chat)

        db.commit()

        db.refresh(new_chat)

        return {
            "response": ai_response
        }

    except Exception as e:

        db.rollback()

        return {
            "error": str(e)
        }

    finally:

        db.close()


@app.post("/create_user")
def create_user(request: UserRequest):
    db = SessionLocal()

    user = User(
        name=request.name,
        email=request.email,
        phone=request.phone
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()

    return {
        "message": "User created",
        "user_id": user.id
    }


@app.post("/create_conversation")
def create_conversation(request: ConversationRequest):

    db = SessionLocal()

    conversation = Conversation(
        user_id=request.user_id
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    db.close()

    return {
        "message": "Conversation created",
        "conversation_id": conversation.id
    }


@app.post("/add_property")
def add_property(request: PropertyRequest):

    db = SessionLocal()

    property_item = Property(
        title=request.title,
        location=request.location,
        price=request.price,
        bhk=request.bhk,
        description=request.description
    )

    db.add(property_item)
    db.commit()
    db.refresh(property_item)
    db.close()

    return {
        "message": "Property added Successfully"
    }


@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)

    text = extract_pdf_text(file_path)

    chunks = chunk_text(text)

    embeddings = create_chunk_embeddings(chunks)

    store_document_chunks(
        chunks,
        embeddings,
        file.filename
    )

    return {
        "message": "PDF processed successfully",
        "total_chunks": len(chunks),

    }
