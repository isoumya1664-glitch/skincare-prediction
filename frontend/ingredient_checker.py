# =========================================================
# INGREDIENT CHECKER
# =========================================================

import re


# =========================================================
# INGREDIENTS THAT REQUIRE ATTENTION
# =========================================================

# These ingredients are not automatically harmful.
# They may cause irritation, dryness, or sensitivity
# for some users depending on skin type and sensitivity.

INGREDIENT_FLAGS = {

    "fragrance": {
        "label": "Fragrance",
        "reason": "May cause irritation or sensitivity in some people.",
        "level": "Caution"
    },

    "parfum": {
        "label": "Parfum",
        "reason": "Fragrance ingredient that may irritate sensitive skin.",
        "level": "Caution"
    },

    "alcohol denat": {
        "label": "Alcohol Denat.",
        "reason": "Can be drying or irritating for some skin types.",
        "level": "Caution"
    },

    "denatured alcohol": {
        "label": "Denatured Alcohol",
        "reason": "Can be drying or irritating for some users.",
        "level": "Caution"
    },

    "isopropyl alcohol": {
        "label": "Isopropyl Alcohol",
        "reason": "May be drying or irritating for sensitive skin.",
        "level": "Caution"
    },

    "menthol": {
        "label": "Menthol",
        "reason": "May cause irritation or sensitivity in some users.",
        "level": "Caution"
    },

    "eucalyptus oil": {
        "label": "Eucalyptus Oil",
        "reason": "May irritate sensitive skin in some people.",
        "level": "Caution"
    },

    "essential oil": {
        "label": "Essential Oils",
        "reason": "Some essential oils may trigger irritation or sensitivity.",
        "level": "Caution"
    },

    "sodium lauryl sulfate": {
        "label": "Sodium Lauryl Sulfate",
        "reason": "Can be irritating or drying for some sensitive skin.",
        "level": "Caution"
    },

    "sodium laureth sulfate": {
        "label": "Sodium Laureth Sulfate",
        "reason": "May be drying or irritating for some users.",
        "level": "Caution"
    }
}


# =========================================================
# KNOWN COSMETIC INGREDIENTS
# =========================================================

KNOWN_INGREDIENTS = [

    "water",
    "aqua",
    "glycerin",
    "niacinamide",
    "salicylic acid",
    "hyaluronic acid",
    "vitamin c",
    "vitamin e",
    "aloe vera",
    "aloe barbadensis leaf extract",
    "ceramide",
    "retinol",
    "fragrance",
    "parfum",
    "alcohol",
    "alcohol denat",
    "denatured alcohol",
    "isopropyl alcohol",
    "ethanol",
    "phenoxyethanol",
    "parabens",
    "benzyl alcohol",
    "sodium benzoate",
    "sodium lauryl sulfate",
    "sodium laureth sulfate",
    "menthol",
    "eucalyptus oil",
    "essential oil"
]


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    # Normalize line breaks and spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# EXTRACT INGREDIENT SECTION
# =========================================================

def extract_ingredient_section(text):

    if not text:
        return ""

    text = str(text)

    text_lower = text.lower()

    keywords = [
        "ingredients:",
        "ingredients",
        "ingredient:",
        "ingredient",
        "composition:",
        "composition",
        "key ingredients:",
        "key ingredients"
    ]

    start = -1

    for keyword in keywords:

        position = text_lower.find(
            keyword
        )

        if position != -1:

            start = (
                position
                + len(keyword)
            )

            break

    # No ingredient heading found
    if start == -1:
        return ""

    ingredient_text = text[start:]

    # Sections that normally come after ingredients
    stop_words = [
        "directions",
        "directions:",
        "how to use",
        "how to use:",
        "warning",
        "warnings",
        "warning:",
        "caution",
        "storage",
        "manufactured by",
        "marketed by",
        "expiry",
        "exp"
    ]

    lower_section = (
        ingredient_text.lower()
    )

    stop_positions = []

    for word in stop_words:

        position = lower_section.find(
            word
        )

        if position != -1:

            stop_positions.append(
                position
            )

    if stop_positions:

        stop = min(
            stop_positions
        )

        ingredient_text = (
            ingredient_text[:stop]
        )

    return ingredient_text.strip()


# =========================================================
# DETECT KNOWN INGREDIENTS
# =========================================================

def detect_ingredients(text):

    if not text:
        return []

    normalized = normalize_text(
        text
    )

    detected = []

    for ingredient in KNOWN_INGREDIENTS:

        if ingredient in normalized:

            # Convert ingredient name
            # into readable format
            label = ingredient.title()

            if label not in detected:

                detected.append(
                    label
                )

    return detected


# =========================================================
# CHECK INGREDIENTS REQUIRING ATTENTION
# =========================================================

def check_ingredients(text):

    if not text:
        return []

    normalized = normalize_text(
        text
    )

    flagged = []

    for ingredient, info in INGREDIENT_FLAGS.items():

        if ingredient in normalized:

            flagged.append(
                {
                    "ingredient": info["label"],
                    "reason": info["reason"],
                    "level": info["level"]
                }
            )

    return flagged


# =========================================================
# COMPLETE INGREDIENT ANALYSIS
# =========================================================

def analyze_ingredients(text):

    if text is None:
        text = ""

    text = str(text).strip()

    if not text:

        return {
            "ingredient_text": "",
            "ingredients": [],
            "flagged": []
        }


    # -----------------------------------------------------
    # 1. Try to extract an Ingredients section
    #    This is mainly useful for OCR text.
    # -----------------------------------------------------

    ingredient_section = (
        extract_ingredient_section(
            text
        )
    )


    # -----------------------------------------------------
    # 2. Decide what text should be analysed
    #
    #    If an "Ingredients:" heading exists:
    #    analyse only that section.
    #
    #    Otherwise:
    #    treat the complete input as the ingredient list.
    #
    #    This is important because your CSV contains
    #    ingredients directly without an "Ingredients:" heading.
    # -----------------------------------------------------

    if ingredient_section:

        text_to_check = (
            ingredient_section
        )

        display_text = (
            ingredient_section
        )

    else:

        text_to_check = text

        display_text = text


    # -----------------------------------------------------
    # 3. Detect known ingredients
    # -----------------------------------------------------

    ingredients = detect_ingredients(
        text_to_check
    )


    # -----------------------------------------------------
    # 4. Detect ingredients requiring attention
    # -----------------------------------------------------

    flagged = check_ingredients(
        text_to_check
    )


    # -----------------------------------------------------
    # 5. Return complete analysis
    # -----------------------------------------------------

    return {

        # IMPORTANT:
        # Never return empty ingredient_text when
        # the input itself is a CSV ingredient list.
        "ingredient_text": display_text,

        "ingredients": ingredients,

        "flagged": flagged
    }