#moovie- recommendation-system 
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Movie Recommendation System\n",
    "\n",
    "Movies will be given by title."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Start by inspecting our dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>movieId</th>\n",
       "      <th>imdbId</th>\n",
       "      <th>tmdbId</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>114709</td>\n",
       "      <td>862.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>113497</td>\n",
       "      <td>8844.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>113228</td>\n",
       "      <td>15602.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>114885</td>\n",
       "      <td>31357.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>113041</td>\n",
       "      <td>11862.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   movieId  imdbId   tmdbId\n",
       "0        1  114709    862.0\n",
       "1        2  113497   8844.0\n",
       "2        3  113228  15602.0\n",
       "3        4  114885  31357.0\n",
       "4        5  113041  11862.0"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "links_df = pd.read_csv('data/links.csv')\n",
    "links_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Jumanji (1995)</td>\n",
       "      <td>Adventure|Children|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Grumpier Old Men (1995)</td>\n",
       "      <td>Comedy|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Waiting to Exhale (1995)</td>\n",
       "      <td>Comedy|Drama|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Father of the Bride Part II (1995)</td>\n",
       "      <td>Comedy</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   movieId                               title  \\\n",
       "0        1                    Toy Story (1995)   \n",
       "1        2                      Jumanji (1995)   \n",
       "2        3             Grumpier Old Men (1995)   \n",
       "3        4            Waiting to Exhale (1995)   \n",
       "4        5  Father of the Bride Part II (1995)   \n",
       "\n",
       "                                        genres  \n",
       "0  Adventure|Animation|Children|Comedy|Fantasy  \n",
       "1                   Adventure|Children|Fantasy  \n",
       "2                               Comedy|Romance  \n",
       "3                         Comedy|Drama|Romance  \n",
       "4                                       Comedy  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_df = pd.read_csv('data/movies.csv')\n",
    "movies_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>userId</th>\n",
       "      <th>movieId</th>\n",
       "      <th>rating</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>964982703</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4.0</td>\n",
       "      <td>964981247</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>4.0</td>\n",
       "      <td>964982224</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>47</td>\n",
       "      <td>5.0</td>\n",
       "      <td>964983815</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>50</td>\n",
       "      <td>5.0</td>\n",
       "      <td>964982931</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   userId  movieId  rating  timestamp\n",
       "0       1        1     4.0  964982703\n",
       "1       1        3     4.0  964981247\n",
       "2       1        6     4.0  964982224\n",
       "3       1       47     5.0  964983815\n",
       "4       1       50     5.0  964982931"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratings_df = pd.read_csv('data/ratings.csv')\n",
    "ratings_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>userId</th>\n",
       "      <th>movieId</th>\n",
       "      <th>tag</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>60756</td>\n",
       "      <td>funny</td>\n",
       "      <td>1445714994</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>60756</td>\n",
       "      <td>Highly quotable</td>\n",
       "      <td>1445714996</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>60756</td>\n",
       "      <td>will ferrell</td>\n",
       "      <td>1445714992</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>89774</td>\n",
       "      <td>Boxing story</td>\n",
       "      <td>1445715207</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>89774</td>\n",
       "      <td>MMA</td>\n",
       "      <td>1445715200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   userId  movieId              tag   timestamp\n",
       "0       2    60756            funny  1445714994\n",
       "1       2    60756  Highly quotable  1445714996\n",
       "2       2    60756     will ferrell  1445714992\n",
       "3       2    89774     Boxing story  1445715207\n",
       "4       2    89774              MMA  1445715200"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tags_df = pd.read_csv('data/tags.csv')\n",
    "tags_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's go on the assumption that if person $A$ likes movies $M_0, M_1, ..., M_i$,\n",
    "then there goes person $B$ who likes one or more movies from $M_i$ let's call them $M_j$.\n",
    "\n",
    "This would mean that $A$ and $B$ has a movie that they both liked, therefore other movies from both $M_j$ and $M_i$ can be liked by both $A$ and $B$ with high probability.\n",
    "\n",
    "------------------"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = movies_df.merge(ratings_df, on='movieId')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Addams Family, The (1991)\n",
      "Batman Forever (1995)\n",
      "Crow, The (1994)\n",
      "Forrest Gump (1994)\n",
      "Heavy Metal (1981)\n",
      "Hellraiser: Bloodline (1996)\n",
      "Indiana Jones and the Temple of Doom (1984)\n",
      "Interview with the Vampire: The Vampire Chronicles (1994)\n",
      "Jerky Boys, The (1995)\n",
      "Major Payne (1995)\n",
      "Maverick (1994)\n",
      "Natural Born Killers (1994)\n",
      "Pan's Labyrinth (Laberinto del fauno, El) (2006)\n",
      "Phenomenon (1996)\n",
      "Pulp Fiction (1994)\n",
      "Santa Clause, The (1994)\n",
      "Secret Garden, The (1993)\n",
      "Silence of the Lambs, The (1991)\n",
      "Tank Girl (1995)\n",
      "Toy Story (1995)\n",
      "True Lies (1994)\n",
      "Willow (1988)\n",
      "Wyatt Earp (1994)\n"
     ]
    }
   ],
   "source": [
    "M_j = 'Jumanji (1995)' # Title as input, now it's just one movie\n",
    "recommended_movies = []\n",
    "\n",
    "# Find the movie in the database, and sort it by rating\n",
    "movie_db = df[df['title'] == M_j]\\\n",
    "            .sort_values(by='rating', ascending=False)\n",
    "\n",
    "# Get the first 5 users who liked this movie\n",
    "for user in movie_db.iloc[:5]['userId'].values:\n",
    "    \n",
    "    # Get the rated movies for this user\n",
    "    rated_movies = df[df['userId'] == user]\n",
    "    \n",
    "    # Get the five biggest rated movie by this user\n",
    "    rated_movies = rated_movies[rated_movies['title'] != M_j]\\\n",
    "                    .sort_values(by='rating', ascending=False)\\\n",
    "                    .iloc[:5]\n",
    "    \n",
    "    # Add these to the recommendations\n",
    "    recommended_movies.extend(list(rated_movies['title'].values))\n",
    "    \n",
    "recommended_movies = np.unique(recommended_movies)\n",
    "    \n",
    "for movie in recommended_movies:\n",
    "    print(movie)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now weight each movie by the similiarity on the genre feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "gmovie_genres = df[df['title'] == M_j].iloc[0]['genres'].split('|')\n",
    "scores = {}  # {title: score ...}\n",
    "\n",
    "for movie in recommended_movies:\n",
    "    movied = df[df['title'] == movie].iloc[0]\n",
    "    movie_genres = movied['genres'].split('|')\n",
    "    score = 0\n",
    "    \n",
    "    # How many gmovie_genre can be found in movie_genres?\n",
    "    for gmovie_genre in gmovie_genres:\n",
    "        if gmovie_genre in movie_genres:\n",
    "            score += 1\n",
    "    \n",
    "    scores[movie] = score\n",
    "    \n",
    "# Sort them on score and reverse it, because the bigger the score the better \n",
    "recommended_movies = sorted(scores, key=lambda x: scores[x])[::-1]  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The recommendations are now weighted"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Toy Story (1995)\n",
      "Willow (1988)\n",
      "Indiana Jones and the Temple of Doom (1984)\n",
      "Addams Family, The (1991)\n",
      "True Lies (1994)\n",
      "Secret Garden, The (1993)\n",
      "Santa Clause, The (1994)\n",
      "Pan's Labyrinth (Laberinto del fauno, El) (2006)\n",
      "Maverick (1994)\n",
      "Heavy Metal (1981)\n",
      "Crow, The (1994)\n",
      "Batman Forever (1995)\n",
      "Wyatt Earp (1994)\n",
      "Tank Girl (1995)\n",
      "Silence of the Lambs, The (1991)\n",
      "Pulp Fiction (1994)\n",
      "Phenomenon (1996)\n",
      "Natural Born Killers (1994)\n",
      "Major Payne (1995)\n",
      "Jerky Boys, The (1995)\n",
      "Interview with the Vampire: The Vampire Chronicles (1994)\n",
      "Hellraiser: Bloodline (1996)\n",
      "Forrest Gump (1994)\n"
     ]
    }
   ],
   "source": [
    "for movie in recommended_movies:\n",
    "    print(movie)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
