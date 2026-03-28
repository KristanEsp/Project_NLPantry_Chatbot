#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import nltk
import spacy
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings('ignore')


#Import the food dataset
df_food = pd.read_csv(os.path.join(BASE_DIR, "dataset", "Recipe_Dataset.csv"))
df_food.tail(10)

text = "i want a recipe with steak"


## Text cleaning
#Perform initial text cleaning: lowercase, punctuations removal, 
def text_preprocessing(text):
    #1.) Lower case the text
    text_lower = text.lower()
    #2.) remove punctuations
    punctuations = r"[!#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"
    text_no_punct = re.sub(punctuations, "", text_lower)
    return text_no_punct
text_no_punct = text_preprocessing(text)

#Tokenize
def perform_tokenization(text_no_punct):
    text_tokenize = word_tokenize(text_no_punct)
    return text_tokenize
    
text_tokenize = perform_tokenization(text_no_punct)

#Remove Stop Word
def perform_stopword_removal(text_tokenize):
    stop_words = set(stopwords.words("english"))
    text_no_stopwords = [word for word in text_tokenize if word not in stop_words]
    return(text_no_stopwords)
    
text_no_stopwords = perform_stopword_removal(text_tokenize)
text_no_stopwords

#lemmatization
def perform_lemmatization(text_no_stopwords):
    wn = nltk.WordNetLemmatizer()
    text_lemmatized = []
    for word in text_no_stopwords:
        text_lemmatized.append(wn.lemmatize(word))
    return text_lemmatized
    
text_lemmatized = perform_lemmatization(text_no_stopwords)
text_lemmatized

#Process tokenization, stop word removal and lemmatization in one function
def nlp_processing(text):
    #Tokenize
    text_tokenization = perform_tokenization(text)
    #Remove Stop Word
    text_nostopwords = perform_stopword_removal(text_tokenization)
    #lemmatization
    text_lemmatized = perform_lemmatization(text_nostopwords)
    #Convert back to string format
    text_cleaned = " ".join(text_lemmatized)
    return text_cleaned, text_lemmatized
text_cleaned,text_token = nlp_processing(text_no_punct)
text_cleaned




# ## Named Entity Recognition for ingredients
# - The default Spacy model does not recognize food items or ingredients
# - So ingredient names must be added to the model
# - This will be done by using the ingredients list (ingredients.txt) and Spacy's PhraseMatcher

#Load the ingredients.txt and load them into a list
def load_ingredients_list():
    with open(os.path.join(BASE_DIR, "dataset", "ingredients.txt"), "r") as file:
        ingredients_list = file.read().split(', ')
        file.close()
    return ingredients_list
    
ingredients_list = load_ingredients_list()
print("Total number of available ingredients: ", len(ingredients_list))


#Using Spacy's PhraseMatcher to tag ingredients in the list
def tag_ingredients(ingredients_list):
    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab, attr = "LOWER")
    #Convert all the ingredients into a doc
    ingredients_doc = []
    #Pre-processess the ingredient text
    for ingredient in ingredients_list:
        #Perform tokenization
        ingredient = word_tokenize(ingredient)
        #Perform lemmatization
        ingredient = perform_lemmatization(ingredient)
        #Convert back to string format
        ingredient = " ".join(ingredient)
        ingredients_doc.append(nlp.make_doc(ingredient))
    #Match all the ingredient names with the "INGREDIENT" tag
    matcher.add("INGREDIENT", ingredients_doc)
    return matcher, nlp

matcher,nlp = tag_ingredients(ingredients_list)


#Create doc and identify ingredients from the text from the matcher tag
def identify_ingredients(text, matcher, nlp):
    doc = nlp(text)
    matches = matcher(doc)
    
    #Scan through the doc to find any ingredients
    ingredient_name = [""]
    for match_id, start, end in matches:
        # Get the tag ID ("Ingredient")
        tag_id = nlp.vocab.strings[match_id]
        # Get the detected ingredient name
        ingredient_id = doc[start:end]
        #Choose the longer/more specific ingredient name - e.g. boneless pork chops instead of pork
        ingredient_name = max(ingredient_name, ingredient_id.text, key = len) 
        return ingredient_name

ingredient_name = identify_ingredients(text_cleaned, matcher, nlp)


# # Matching Ingredients to recipes
# - Use TFIDF to convert both the user's ingredients list and the dataset ingredient list into a vector
# - Compare those vector using cosine_similarity

#Use TFIDF to vectorize the dataset's ingredients list
def perform_tfidf(ingredients_dataset, ingredients_list):
    tfidf = TfidfVectorizer(lowercase = True, vocabulary = ingredients_list, min_df = 1)
    df_vector = tfidf.fit_transform(ingredients_dataset)
    return tfidf, df_vector

df_tfidf, df_vector = perform_tfidf(df_food["ingredients"], ingredients_list)
feature_names = df_tfidf.get_feature_names_out()
feature_names[-10:]


#Use cosine similarity to compare the vectors
def perform_cosine_similarity(df_tfidf, df_vector, user_ingredients):
    user_vector = df_tfidf.transform(user_ingredients)
    similarity = cosine_similarity(user_vector, df_vector).flatten()
    
    #Sort the recipe based on cosine similarity score
    sorted_recipe = np.argsort(similarity)[::-1] #sort by descending
    return sorted_recipe