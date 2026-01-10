#code here
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Sample Movie Dataset
# -----------------------------
data = {
    "title": [
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Avatar",
        "Titanic",
        "The Matrix",
        "John Wick",
        "Avengers",
        "Iron Man",
        "Doctor Strange"
    ],
    "description": [
        "dream subconscious mind thriller",
        "space time black hole science fiction",
        "batman joker crime action",
        "alien planet sci fi adventure",
        "romantic tragedy ship",
        "virtual reality hacking sci fi",
        "assassin revenge action thriller",
        "superheroes action sci fi",
        "technology superhero action",
        "magic superhero multiverse"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["description"])

# -----------------------------
# Similarity Matrix
# -----------------------------
similarity = cosine_similarity(tfidf_matrix)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie_name, top_n=5):
    if movie_name not in df["title"].values:
        return "Movie not found!"

    idx = df[df["title"] == movie_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in scores[1:top_n+1]:
        recommendations.append(df.iloc[i[0]]["title"])

    return recommendations


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    movie = input("Enter a movie name: ")
    print("\nRecommended Movies:")
    print(recommend(movie))
