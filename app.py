import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ML Internship Dashboard",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ ML Internship Project Dashboard")
st.caption("Unified Machine Learning Application | Dark Mode UI")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
module = st.sidebar.radio(
    "Select ML Module",
    [
        "⭐ Restaurant Rating Prediction",
        "🍴 Restaurant Recommendation",
        "📊 Cuisine Classification"
    ]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_raw_data():
    return pd.read_csv("Dataset .csv", encoding="latin1")

@st.cache_data
def load_cleaned_data():
    return pd.read_csv("cleaned_data.csv")

raw_df = load_raw_data()
clean_df = load_cleaned_data()

# ==================================================
# MODULE 1: RESTAURANT RATING PREDICTION
# ==================================================
if module == "⭐ Restaurant Rating Prediction":

    st.subheader("⭐ Restaurant Rating Prediction")
    st.write("Predict restaurant ratings using Random Forest Regression.")

    feature_cols = [
        "Country Code", "City", "Cuisines",
        "Average Cost for two", "Has Table booking",
        "Has Online delivery", "Price range", "Votes"
    ]

    X = clean_df[feature_cols]
    y = clean_df["Aggregate rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country Code", sorted(clean_df["Country Code"].unique()))
        city = st.selectbox("City (Encoded)", sorted(clean_df["City"].unique()))
        cuisine = st.selectbox("Cuisine (Encoded)", sorted(clean_df["Cuisines"].unique()))
        cost = st.slider("Average Cost for Two", 100, 5000, 500)

    with col2:
        table = st.radio("Table Booking", [0, 1], format_func=lambda x: "Yes" if x else "No")
        delivery = st.radio("Online Delivery", [0, 1], format_func=lambda x: "Yes" if x else "No")
        price = st.selectbox("Price Range", sorted(clean_df["Price range"].unique()))
        votes = st.slider("Votes", 0, int(clean_df["Votes"].max()), 100)

    if st.button("Predict Rating"):
        # ✅ FIX: Use DataFrame with feature names
        input_df = pd.DataFrame([{
            "Country Code": country,
            "City": city,
            "Cuisines": cuisine,
            "Average Cost for two": cost,
            "Has Table booking": table,
            "Has Online delivery": delivery,
            "Price range": price,
            "Votes": votes
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"⭐ Predicted Rating: **{prediction:.2f} / 5.0**")

# ==================================================
# MODULE 2: RESTAURANT RECOMMENDATION SYSTEM
# ==================================================
elif module == "🍴 Restaurant Recommendation":

    st.subheader("🍴 Restaurant Recommendation System")
    st.write("Content-based recommendation using TF-IDF and Cosine Similarity.")

    df = raw_df[
        ['Restaurant Name', 'City', 'Locality',
         'Cuisines', 'Average Cost for two',
         'Aggregate rating', 'Votes']
    ].dropna()

    df['content'] = (
        df['Cuisines'].astype(str) + " " +
        df['City'].astype(str) + " " +
        df['Locality'].astype(str)
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tf_matrix = vectorizer.fit_transform(df['content'])

    cuisine = st.text_input("Preferred Cuisine", "North Indian")
    city = st.text_input("City", "Bangalore")
    budget = st.slider("Maximum Budget", 100, 5000, 600)

    if st.button("Recommend Restaurants"):
        user_vec = vectorizer.transform([cuisine + " " + city])
        similarity = cosine_similarity(user_vec, tf_matrix).flatten()

        temp_df = df.copy()
        temp_df["similarity"] = similarity

        filtered = temp_df[
            (temp_df["City"].str.lower() == city.lower()) &
            (temp_df["Average Cost for two"] <= budget)
        ]

        if filtered.empty:
            st.warning("No restaurants found for the given preferences.")
        else:
            result = filtered.sort_values(
                by=["similarity", "Aggregate rating", "Votes"],
                ascending=False
            ).head(5)

            st.dataframe(result.reset_index(drop=True))

# ==================================================
# MODULE 3: CUISINE CLASSIFICATION
# ==================================================
elif module == "📊 Cuisine Classification":

    st.subheader("📊 Cuisine Classification")
    st.write("Predict cuisine category using a trained ML classifier.")

    @st.cache_resource
    def train_cuisine_model():
        df = raw_df.copy()
        df["Cuisines"].fillna("Other", inplace=True)

        def map_cuisine(c):
            c = str(c).lower()
            if "indian" in c or "biryani" in c:
                return "Indian"
            elif "chinese" in c or "asian" in c:
                return "Chinese"
            elif "pizza" in c or "pasta" in c:
                return "Italian"
            elif "burger" in c or "fast food" in c:
                return "Fast Food"
            elif "cafe" in c or "coffee" in c:
                return "Cafe"
            else:
                return "Other"

        df["Cuisine_Group"] = df["Cuisines"].apply(map_cuisine)

        le = LabelEncoder()
        df["Cuisine_Label"] = le.fit_transform(df["Cuisine_Group"])

        features = ["Average Cost for two", "Votes", "Aggregate rating"]
        df[features] = df[features].fillna(df[features].mean())

        X = df[features]
        y = df["Cuisine_Label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        model = GradientBoostingClassifier(
            n_estimators=300,        # 🔽 reduced for speed
            learning_rate=0.05,
            random_state=42
        )

        model.fit(X_train, y_train)

        return model, le, features

    model, label_encoder, feature_cols = train_cuisine_model()

    cost = st.slider("Average Cost for Two", 100, 5000, 400)
    votes = st.slider("Votes", 0, 10000, 200)
    rating = st.slider("Aggregate Rating", 1.0, 5.0, 4.0)

    if st.button("Predict Cuisine"):
        input_df = pd.DataFrame([{
            "Average Cost for two": cost,
            "Votes": votes,
            "Aggregate rating": rating
        }])

        pred = model.predict(input_df)[0]
        cuisine = label_encoder.inverse_transform([pred])[0]

        st.success(f"🍽️ Predicted Cuisine Category: **{cuisine}**")
