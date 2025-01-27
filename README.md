User-Based Collaborative Filtering in Python | Machine Learning

This repository contains an implementation of the User-Based Collaborative Filtering recommendation system using Python. The system suggests movies to users based on the ratings of similar users. It applies various machine learning techniques, such as data cleaning, exploratory data analysis (EDA), user-movie matrix creation, and similarity measurement to provide personalized recommendations.

Table of Contents

Introduction
Project Steps
Technologies Used
Installation
How to Run
Conclusion
Introduction

Collaborative filtering is a popular technique used in recommendation systems. The User-Based Collaborative Filtering algorithm makes recommendations based on the idea that users who have rated items similarly in the past will rate items similarly in the future.

This project walks you through the process of building a user-based collaborative filtering model using Python. It includes all the steps from importing necessary libraries, reading the data, performing exploratory data analysis, to providing movie recommendations.

Project Steps

0: User-Based Collaborative Filtering Recommendation Algorithm
Introduction to the concept of User-Based Collaborative Filtering and how the algorithm works. The recommendation is made by calculating the similarity between users based on their rating patterns and recommending items rated highly by similar users.
1: Import Python Libraries
Libraries like Pandas, NumPy, Matplotlib, and Seaborn are used for data manipulation and visualization.
Machine learning libraries like Scikit-learn can be used for data processing and similarity calculation.
2: Download and Read in Data
In this step, we load the dataset containing movie ratings by users. The dataset can be downloaded from various sources, such as MovieLens or custom datasets.
import pandas as pd

# Read in data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
3: Exploratory Data Analysis (EDA)
The EDA phase involves inspecting the dataset to check for missing values, understand the distribution of data, and identify patterns.
# Basic exploration
ratings.info()
ratings.describe()
4: Create User-Movie Matrix
The user-movie matrix is created by pivoting the ratings data, where users are rows and movies are columns.
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
5: Data Normalization
Data normalization ensures that the ratings of each user are transformed in a way that improves similarity calculations.
normalized_matrix = user_movie_matrix.sub(user_movie_matrix.mean(axis=1), axis=0)
6: Identify Similar Users
We calculate user similarity using a metric such as Pearson Correlation or Cosine Similarity to find users with similar rating patterns.
user_similarity = normalized_matrix.T.corr()
7: Narrow Down Item Pool
By using the similarities between users, we narrow down the item pool (movies) to the most relevant ones for each user. This reduces computation and increases recommendation quality.
similar_users = user_similarity[user_similarity > 0.5]
8: Recommend Items
After identifying similar users and narrowing the item pool, we recommend items (movies) that similar users have rated highly.
item_scores = {}
# Calculate item scores based on similar users' ratings
Technologies Used

Python - Programming Language
Pandas - Data manipulation and analysis
NumPy - Numerical computation
Matplotlib & Seaborn - Data visualization
Scikit-learn - For machine learning algorithms and metrics (optional)
Jupyter Notebook - For interactive coding and exploration
