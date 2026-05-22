

def detect_intent(message: str):

    message_lower = message.lower()

    property_keywords = [
        "property",
        "apartment",
        "flat",
        "bhk",
        "price",
        "rent",
        "buy"
    ]

    document_keywords = [
        "pdf",
        "document",
        "brochure",
        "amenties",
        "location",
        "details"
    ]

    for keyword in property_keywords:

        if keyword in message_lower:

            return "property_search"

    for keyword in document_keywords:

        if keyword in message_lower:

            return "document_search"

        return "general"


def decide_tools(intent):

    if intent == "property_search":

        return [
            "postgres_search",
            "semantic_property_search"
        ]

    elif intent == "document_search":

        return [
            "chroma_rag_search"
        ]

    return [
        "llm_only"
    ]
