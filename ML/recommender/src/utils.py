import re


def normalize_text(value):

    if value is None:
        return ""

    value = str(value).lower().strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


def clean_price(price):

    if price is None:
        return "Price unavailable"

    price = str(price).strip()

    if not price:
        return "Price unavailable"

    return price


def get_product_name(row):

    if "name" in row.index:

        name = str(
            row["name"]
        ).strip()

        if name:
            return name

    return "Unknown Product"


def get_product_brand(row):

    if "brand" in row.index:

        brand = str(
            row["brand"]
        ).strip()

        if brand:
            return brand

    return "Unknown Brand"


def get_product_price(row):

    if "price" in row.index:

        return clean_price(
            row["price"]
        )

    return "Price unavailable"


def get_product_category(row):

    if "label" in row.index:

        category = str(
            row["label"]
        ).strip()

        if category:
            return category

    return "Unknown Category"


def get_product_skin_type(row):

    if "skin type" in row.index:

        skin_type = str(
            row["skin type"]
        ).strip()

        if skin_type:
            return skin_type

    return "Not specified"


def get_product_concerns(row):

    concerns = []

    for column in [
        "concern",
        "concern 2",
        "concern 3"
    ]:

        if column in row.index:

            value = str(
                row[column]
            ).strip()

            if value:
                concerns.append(
                    value
                )

    return concerns


def get_product_ingredient(row):

    if "key ingredient" in row.index:

        ingredient = str(
            row["key ingredient"]
        ).strip()

        if ingredient:
            return ingredient

    return "Not specified"


def get_product_formulation(row):

    if "formulation" in row.index:

        formulation = str(
            row["formulation"]
        ).strip()

        if formulation:
            return formulation

    return "Not specified"


def get_product_url(row):

    if "url" in row.index:

        url = str(
            row["url"]
        ).strip()

        if url.startswith("http"):
            return url

    return ""