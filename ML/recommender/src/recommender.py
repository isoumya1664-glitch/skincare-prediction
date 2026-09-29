from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"


# ---------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


def split_values(value):

    if not value:
        return []

    value = normalize_text(value)

    values = re.split(
        r"[,;/|]+",
        value
    )

    return [
        item.strip()
        for item in values
        if item.strip()
    ]


# ---------------------------------------------------------
# RECOMMENDER CLASS
# ---------------------------------------------------------

class SkinCareRecommender:

    def __init__(
        self,
        dataset_path=None
    ):

        if dataset_path is None:

            dataset_path = (
                DATA_DIR /
                "general_skincare.csv"
            )

        self.dataset_path = Path(
            dataset_path
        )

        self.df = self._load_data()

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    def _load_data(self):

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: "
                f"{self.dataset_path}"
            )

        df = pd.read_csv(
            self.dataset_path
        )

        df.columns = [
            normalize_text(column)
            for column in df.columns
        ]

        # Remove unwanted columns
        df = df.drop(
            columns=[
                "unnamed: 0",
                "index"
            ],
            errors="ignore"
        )

        # Fill missing values
        for column in df.columns:

            if df[column].dtype == "object":

                df[column] = (
                    df[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

        return df

    # -----------------------------------------------------
    # SKIN TYPE MATCH
    # -----------------------------------------------------

    def skin_type_match(
        self,
        product_skin_type,
        user_skin_type
    ):

        product_skin_type = normalize_text(
            product_skin_type
        )

        user_skin_type = normalize_text(
            user_skin_type
        )

        if product_skin_type in [
            "",
            "all",
            "all skin types"
        ]:
            return True

        return (
            user_skin_type
            in product_skin_type
        )

    # -----------------------------------------------------
    # CONCERN MATCH
    # -----------------------------------------------------

    def concern_match(
        self,
        row,
        user_concerns
    ):

        if not user_concerns:
            return 0

        product_concerns = []

        for column in [
            "concern",
            "concern 2",
            "concern 3"
        ]:

            if column in row.index:

                product_concerns.extend(
                    split_values(
                        row[column]
                    )
                )

        score = 0

        for user_concern in user_concerns:

            user_concern = normalize_text(
                user_concern
            )

            for product_concern in product_concerns:

                if (
                    user_concern in product_concern
                    or
                    product_concern in user_concern
                ):
                    score += 1
                    break

        return score

    # -----------------------------------------------------
    # RECOMMEND
    # -----------------------------------------------------

    def recommend(
        self,
        skin_type,
        concerns=None,
        category=None,
        limit=8
    ):

        if concerns is None:
            concerns = []

        concerns = [
            normalize_text(c)
            for c in concerns
        ]

        df = self.df.copy()

        # ---------------------------------------------
        # CATEGORY FILTER
        # ---------------------------------------------

        if (
            category
            and "label" in df.columns
        ):

            category_normalized = (
                normalize_text(category)
            )

            category_df = df[
                df["label"]
                .apply(normalize_text)
                == category_normalized
            ]

            if not category_df.empty:
                df = category_df

        # ---------------------------------------------
        # SKIN TYPE
        # ---------------------------------------------

        if "skin type" in df.columns:

            skin_df = df[
                df["skin type"].apply(
                    lambda x:
                    self.skin_type_match(
                        x,
                        skin_type
                    )
                )
            ]

            if not skin_df.empty:
                df = skin_df

        # ---------------------------------------------
        # CONCERN SCORE
        # ---------------------------------------------

        df["_concern_score"] = df.apply(
            lambda row:
            self.concern_match(
                row,
                concerns
            ),
            axis=1
        )

        # ---------------------------------------------
        # SORT
        # ---------------------------------------------

        df = df.sort_values(
            by="_concern_score",
            ascending=False
        )

        # ---------------------------------------------
        # REMOVE INTERNAL SCORE
        # ---------------------------------------------

        results = df.head(
            limit
        ).copy()

        results = results.drop(
            columns=[
                "_concern_score"
            ],
            errors="ignore"
        )

        return results

    # -----------------------------------------------------
    # GET PRODUCT DETAILS
    # -----------------------------------------------------

    def get_product_details(
        self,
        product_name
    ):

        if "name" not in self.df.columns:
            return None

        target = normalize_text(
            product_name
        )

        matches = self.df[
            self.df["name"]
            .apply(normalize_text)
            == target
        ]

        if matches.empty:
            return None

        return matches.iloc[0]


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    recommender = SkinCareRecommender()

    results = recommender.recommend(
        skin_type="Oily",
        concerns=[
            "Acne",
            "Blackheads"
        ],
        limit=5
    )

    print("\nRecommended Products:\n")

    print(results)