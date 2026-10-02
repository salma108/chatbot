# app.py
import logging
import os
import random
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===========================
# Logging
# ===========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===========================
# Charger clé API
# ===========================
load_dotenv()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
FALLBACK_GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

# ===========================
# Config
# ===========================
TOP_K = 100
MAX_RETRIES = 5
RETRY_DELAY = 1
MAX_CONTEXT_PROFILES = 50

# ===========================
# Charger CSV
# ===========================
@st.cache_data
def load_profiles(csv_path="profiles.csv"):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return df

        df = df.fillna("")
        df["texte"] = (
            df["Nom"].astype(str)
            + " | "
            + df["Competences"].astype(str)
            + " | "
            + df["Localisation"].astype(str)
            + " | "
            + df["Disponibilite"].astype(str)
            + " | Exp: "
            + df["Experience (annees)"].astype(str)
            + " ans"
        )
        return df
    except Exception as exc:
        logging.exception("Impossible de charger le fichier profiles.csv")
        return pd.DataFrame()


df = load_profiles()
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
X = vectorizer.fit_transform(df["texte"]) if not df.empty else None


def retrieve_profiles(query, top_k=TOP_K):
    query = (query or "").strip()
    if not query or X is None or df.empty:
        return pd.DataFrame()

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X).flatten()

    results = df.copy()
    results["score"] = sims
    top_idx = sims.argsort()[::-1][: min(top_k, len(results))]
    return results.iloc[top_idx].reset_index(drop=True)


def build_local_summary(profiles):
    if profiles.empty:
        return "Aucun profil disponible."

    lines = []
    for _, row in profiles.head(5).iterrows():
        lines.append(
            f"- {row['Nom']} | Compétences: {row['Competences']} | Localisation: {row['Localisation']} | "
            f"Expérience: {row['Experience (annees)']} ans | Disponibilité: {row['Disponibilite']} | "
            f"Motif: correspondance forte à la demande."
        )
    return "\n".join(lines)


def resolve_groq_model(client):
    preferred_model = (os.getenv("GROQ_MODEL") or GROQ_MODEL or "").strip()
    candidates = []
    for model_name in [preferred_model, *FALLBACK_GROQ_MODELS]:
        if model_name and model_name not in candidates:
            candidates.append(model_name)

    for model_name in candidates:
        try:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return model_name
        except Exception:
            continue

    return preferred_model or FALLBACK_GROQ_MODELS[0]


def generate_rag_response(profiles, query):
    if profiles.empty:
        return "Aucun profil disponible."

    if not GROQ_API_KEY:
        return build_local_summary(profiles)

    context = "\n".join(profiles["texte"].drop_duplicates().tolist()[:MAX_CONTEXT_PROFILES])
    prompt = f"""
Voici les profils freelances récupérés :
{context}

Requête utilisateur : "{query}"

Tâches :
1. Sélectionne les 3-5 profils les plus pertinents.
2. Donne une justification concise (1-2 phrases).
3. Format : Nom, Compétences, Localisation, Expérience, Disponibilité, Justification.
"""

    client = Groq(api_key=GROQ_API_KEY)
    active_model = resolve_groq_model(client)
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant intelligent qui sélectionne les profils freelances les plus pertinents.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=active_model,
                max_tokens=600,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait_time = (RETRY_DELAY * 2**retries) + random.uniform(0, 1)
            time.sleep(wait_time)
            retries += 1
        except Exception as exc:
            logging.exception("Erreur lors de la requête LLM")
            return f"Erreur lors de la requête LLM : {exc}. Retour à la recommandation locale.\n\n{build_local_summary(profiles)}"

    return "Désolé, le service LLM est temporairement indisponible.\n\n" + build_local_summary(profiles)


# ===========================
# Interface Streamlit
# ===========================
st.set_page_config(page_title="Recommandation Freelance", layout="wide")

st.title("Assistant Intelligent de Recommandation de Profils")

if not GROQ_API_KEY:
    st.warning("Aucune clé API GROQ n’a été détectée. La recommandation fonctionne avec une logique locale de correspondance.")

if df.empty:
    st.error("Aucun profil chargé. Vérifie le fichier profiles.csv.")
else:
    query = st.text_input("Entrez votre recherche :", "")
    if query:
        with st.spinner("Recherche des profils les plus pertinents..."):
            top_profiles = retrieve_profiles(query, TOP_K)
            output = generate_rag_response(top_profiles, query)

        st.subheader("Résultats de la Recommandation")
        st.write(output)

        with st.expander("Profils analysés"):
            st.dataframe(
                top_profiles[["Nom", "Competences", "Localisation", "Disponibilite", "Experience (annees)", "score"]]
            )
    else:
        st.info("Saisissez un besoin métier, une compétence, une localisation ou une disponibilité pour obtenir des recommandations.")
