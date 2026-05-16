from app.models.property_model import Property


def search_properties(db, location=None, bhk=None, max_price=None):

    query = db.query(Property)

    if location:

        query = query.filter(
            Property.location.ilike(f"%{location}")
        )

        if bhk:
            query = query.filter(
                Property.bhk == bhk
            )

        if max_price:

            query = query.filter(
                Property.price <= max_price
            )

    return query.all()
