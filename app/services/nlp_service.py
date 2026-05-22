import re


def extract_preferences(message: str):

    message_lower = message.lower()

    location = None
    bhk = None
    budget = None

    locations = [
        "mumbai",
        "pune",
        "delhi",
        "bangalore"
    ]

    for city in locations:
        if city in message_lower:
            location = city.title()

    bhk_match = re.search(r'(\d+)\s*bhk', message_lower)

    if bhk_match:
        bhk = int(bhk_match.group(1))

    budget_match = re.search(r'(\d+)\s*(lakh|lakh|crore)', message_lower)

    if budget_match:

        amount = int(budget_match.group(1))

        unit = budget_match.group(2)

        if "crore" in unit:
            budget = amount * 10000000

        else:
            budget = amount * 100000

    return {
        "location": location,
        "bhk": bhk,
        "budget": budget
    }
