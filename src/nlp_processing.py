#!/usr/bin/env python
# coding: utf-8

# In[48]:


# get_ipython().system('pip install spacy')
# get_ipython().system('python -m spacy download en_core_web_sm')
# !pip install ingredient-parser-nlp


# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
import spacy
nltk.download('punkt_tab')
# from gensim.models import Doc2Vec
# from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Import the food dataset
df_food = pd.read_csv(os.path.join(BASE_DIR, "dataset", "Recipe_Dataset.csv"))
df_food.tail(10)


# In[3]:


###########
text = "i want a recipe with mixed nuts"


# # Text cleaning

# In[4]:


#Perform initial text cleaning: lowercase, punctuations removal, 
def text_preprocessing(text):
    #1.) Lower case the text
    text_lower = text.lower()
    #2.) remove punctuations
    punctuations = r"[!#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"
    text_no_punct = re.sub(punctuations, "", text_lower)
    return text_no_punct
text_no_punct = text_preprocessing(text)


# In[5]:


#Tokenize
def perform_tokenization(text_no_punct):
    text_tokenize = word_tokenize(text_no_punct)
    return text_tokenize
    
text_tokenize = perform_tokenization(text_no_punct)


# In[6]:


#Remove Stop Word
def perform_stopword_removal(text_tokenize):
    stop_words = set(stopwords.words("english"))
    text_no_stopwords = [word for word in text_tokenize if word not in stop_words]
    return(text_no_stopwords)
    
text_no_stopwords = perform_stopword_removal(text_tokenize)
text_no_stopwords


# In[7]:


#lemmatization
def perform_lemmatization(text_no_stopwords):
    wn = nltk.WordNetLemmatizer()
    text_lemmatized = []
    for word in text_no_stopwords:
        text_lemmatized.append(wn.lemmatize(word))
    return text_lemmatized
    
text_lemmatized = perform_lemmatization(text_no_stopwords)
text_lemmatized


# In[8]:


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
# 

# - The default Spacy model does not recognize food items or ingredients
# - So ingredient names must be added to the model
# - This will be done by using the ingredients list (ingredients.txt) and Spacy's PhraseMatcher

# In[13]:


#Load the ingredients.txt and load them into a list
def load_ingredients_list():
    with open(os.path.join(BASE_DIR, "dataset", "ingredients.txt"), "r") as file:
        ingredients_list = file.read().split(', ')
        file.close()
    return ingredients_list
    
ingredients_list = load_ingredients_list()
print("Total number of available ingredients: ", len(ingredients_list))


# In[10]:


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


# In[11]:


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
print("Detected Ingredient:", ingredient_name)


# # Matching Ingredients to recipes

# - Use TFIDF to convert both the user's ingredients list and the dataset ingredient list into a vector
# - Compare those vector using cosine_similarity

# In[12]:



#Use TFIDF to vectorize the dataset's ingredients list
def perform_tfidf(ingredients_dataset):
    tfidf = TfidfVectorizer(lowercase = True, analyzer = "word", ngram_range = (1,1), norm = 'l1')
    df_vector= tfidf.fit_transform(ingredients_dataset)
    return tfidf, df_vector

df_tfidf, df_vector = perform_tfidf(df_food["ingredients"])
feature_names = df_tfidf.get_feature_names_out()
feature_names[-10:]


# In[34]:


#Use cosine similarity to compare the vectors
def perform_cosine_similarity(df_tfidf, df_vector, user_ingredients):
    user_vector = df_tfidf.transform(user_ingredients)
    similarity = cosine_similarity(user_vector, df_vector).flatten()
    
    #Get the top 5 matching recipies
    top_5 = np.argsort(similarity)[::-1][:5] #sort by descending and get the first 5
    return top_5

user_ingredients = ["chicken, gravy, pea, potato"]
top_5 = perform_cosine_similarity(df_tfidf, df_vector, user_ingredients)
top_5 = df_food["name"][top_5]
print(top_5)


# In[14]:


# ######Get topx
# top_1 = np.argsort(similarity)[::-1][4] #sort by descending and get the first 5
# top_1 = df_food["ingredients"][top_1]
# top_1


# # Test for other functions

# In[29]:


# #Using Blaeu score
# from nltk.translate.bleu_score import sentence_bleu
# reference = [['Paella']]
# candidate = ['Simple', 'Paella']
# score = sentence_bleu(reference, candidate)
# print("BLUE Score:", score)
# df_food["ingredients"][1]


# In[16]:


# # Using Word2 net
# from nltk.corpus import wordnet as wn

# word1 = "done"
# word2 = "finish"

# #Get a list of words that are synonymous with the original word
# synonyms = []
# for synonym in wn.synsets(word2):
#     for lemma in synonym.lemma_names():
#         synonyms.append(lemma)

# synonyms.append("done")
# print(synonyms)


# In[17]:


# from gensim.models import KeyedVectors
# from gensim.downloader import load

# glove_model = load("glove-wiki-gigaword-50")
# word_pairs = [("done", "finish"), ("completed")]

# #Compute similarity
# for pair in word_pairs:
#     similarity = glove_model.similarity(pair[0], pair[1])
# print(similarity)



# #doc2vec
# # define a list of documents.
# data = ["This is the first document",
#         "This is the second document",
#         "This is the third document",
#         "This is the fourth document"]

# #Transform the dataframe's ingredients list into documents
# df_doc = []
# for i, doc in enumerate(df_food["ingredients"]):
#     #Clean the text by tokenizing and lowercase
#     tokenized = word_tokenize(doc.lower())
#     tags = [str(i)]
#     doc = TaggedDocument(tokenized, tags)
#     df_doc.append(doc)

# #Train the Doc2vec model
# model = Doc2Vec(vector_size = 20, min_count = 2, epochs = 50)
# model.build_vocab(df_doc)
# model.train(df_doc, total_examples = model.corpus_count, epochs = model.epochs)

# #Get the document vectors
# df_vector = [model.infer_vector(
#     word_tokenize(doc.lower())) for doc in df_food["ingredients"]]
# user_ingredients = ["gravy, pea, potato, steak"]
# user_vector = [model.infer_vector(
#     word_tokenize(doc.lower())) for doc in user_ingredients]


# #Use cosine similarity to compare the vectors
# def perform_cosine_similarity(user_vector, df_vector):
#     similarity = cosine_similarity(user_vector, df_vector).flatten()
    
#     #Get the top 5 matching recipies
#     top_5 = np.argsort(similarity)[::-1][:5] #sort by descending and get the first 5
#     return top_5


# top_5 = perform_cosine_similarity(user_vector, df_vector)
# top_5 = df_food["name"][top_5]
# print(top_5)



# #  print the document vectors
# for i, doc in enumerate(df_food["ingredients"]):
#     print("Document", i+1, ":", doc)
#     print("Vector:", document_vectors[i])
#     print()