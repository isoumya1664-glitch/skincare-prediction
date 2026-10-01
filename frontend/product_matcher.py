# =========================================================
# PRODUCT MATCHER
# =========================================================

import os
import pandas as pd
import torch

from PIL import Image

from transformers import (
    CLIPProcessor,
    CLIPModel
)

from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

CSV_PATH = os.path.join(
    PROJECT_DIR,
    "ML",
    "recommender",
    "data",
    "processed",
    "makeup_final.csv"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "product_images"
)


# =========================================================
# LOAD CSV
# =========================================================

print("\nLoading product CSV...")

df = pd.read_csv(
    CSV_PATH
)

# Normalize column names
df.columns = [
    str(col).strip().lower()
    for col in df.columns
]

print(
    "CSV columns:",
    df.columns.tolist()
)


# =========================================================
# INGREDIENTS COLUMN
# =========================================================

if "ingredients" not in df.columns:

    print(
        "WARNING: ingredients column "
        "not found in CSV."
    )

    df["ingredients"] = ""

else:

    df["ingredients"] = (
        df["ingredients"]
        .fillna("")
        .astype(str)
    )

    print(
        "Ingredients column found."
    )


# =========================================================
# IMAGE MAPPING
# =========================================================

# Your images are stored separately:
#
# 001.png
# 002.png
# 003.png
# 004.png
# 005.png
#
# So we map the first five CSV products
# to these five images.

image_files = [
    "001.png",
    "002.png",
    "003.png",
    "004.png",
    "005.png"
]


# Create empty image column
df["image"] = ""


# Assign images to first five products
for i in range(
    min(len(df), len(image_files))
):

    df.iloc[
        i,
        df.columns.get_loc("image")
    ] = image_files[i]


print("\nImage mapping:")

for i in range(
    min(len(df), 5)
):

    print(
        df.iloc[i]["name"]
        if "name" in df.columns
        else f"Product {i + 1}",
        "->",
        df.iloc[i]["image"]
    )


# =========================================================
# KEEP ONLY PRODUCTS WITH IMAGES
# =========================================================

df = df[
    df["image"].notna()
    &
    (
        df["image"]
        .astype(str)
        .str.strip()
        != ""
    )
].copy()


# =========================================================
# LOAD CLIP
# =========================================================

print("\nLoading CLIP model...")

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()

print(
    "CLIP model loaded."
)


# =========================================================
# IMAGE EMBEDDING
# =========================================================

def get_image_embedding(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.get_image_features(
            **inputs
        )

        # Handle different CLIP output formats
        if hasattr(
            output,
            "pooler_output"
        ):

            output = output.pooler_output

        elif hasattr(
            output,
            "image_embeds"
        ):

            output = output.image_embeds

        elif hasattr(
            output,
            "last_hidden_state"
        ):

            output = (
                output.last_hidden_state[:, 0, :]
            )

    # Normalize embedding
    norm = output.norm(
        p=2,
        dim=-1,
        keepdim=True
    )

    embedding = (
        output /
        (norm + 1e-8)
    )

    return embedding.cpu().numpy()


# =========================================================
# CREATE PRODUCT EMBEDDINGS
# =========================================================

product_embeddings = []

valid_rows = []


for index, row in df.iterrows():

    image_name = str(
        row["image"]
    ).strip()

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    if not os.path.exists(
        image_path
    ):

        print(
            f"Image not found: {image_name}"
        )

        continue

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        embedding = get_image_embedding(
            image
        )

        product_embeddings.append(
            embedding[0]
        )

        valid_rows.append(
            index
        )

        print(
            f"Processed {image_name}"
        )

    except Exception as e:

        print(
            f"Error processing "
            f"{image_name}: {e}"
        )


# =========================================================
# CHECK EMBEDDINGS
# =========================================================

if not product_embeddings:

    raise RuntimeError(
        "No product images were processed. "
        "Check frontend/product_images/"
    )


product_embeddings = torch.tensor(
    product_embeddings
).numpy()


print(
    f"\nLoaded {len(valid_rows)} "
    f"product images."
)


# =========================================================
# FIND PRODUCT
# =========================================================

def find_product(uploaded_file):

    # -----------------------------------------------------
    # Read uploaded image
    # -----------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # -----------------------------------------------------
    # Create uploaded image embedding
    # -----------------------------------------------------

    uploaded_embedding = (
        get_image_embedding(image)
    )


    # -----------------------------------------------------
    # Compare with catalogue images
    # -----------------------------------------------------

    similarities = cosine_similarity(
        uploaded_embedding,
        product_embeddings
    )[0]


    # -----------------------------------------------------
    # Find best match
    # -----------------------------------------------------

    best_position = similarities.argmax()

    best_csv_index = (
        valid_rows[
            best_position
        ]
    )


    # -----------------------------------------------------
    # Get complete product row
    # -----------------------------------------------------

    product = (
        df.loc[
            best_csv_index
        ]
        .to_dict()
    )


    # -----------------------------------------------------
    # Similarity
    # -----------------------------------------------------

    product["similarity"] = float(
        similarities[
            best_position
        ]
    )


    # -----------------------------------------------------
    # DEBUG
    # -----------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "MATCHED PRODUCT"
    )

    print(
        "=============================="
    )

    print(
        "Name:",
        product.get(
            "name",
            "N/A"
        )
    )

    print(
        "Image:",
        product.get(
            "image",
            "N/A"
        )
    )

    print(
        "Ingredients:",
        product.get(
            "ingredients",
            "NOT FOUND"
        )
    )

    print(
        "Similarity:",
        product["similarity"]
    )

    print(
        "==============================\n"
    )


    # -----------------------------------------------------
    # Similarity threshold
    # -----------------------------------------------------

    if product["similarity"] < 0.55:

        return {
            "status": "Product not found",
            "similarity": product[
                "similarity"
            ]
        }


    # -----------------------------------------------------
    # Product found
    # -----------------------------------------------------

    product["status"] = (
        "Product found"
    )

    return product