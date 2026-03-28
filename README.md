<h1 align="center">NLPantry</h1>

# Introduction
- NLPantry is a simple dialogue chatbot system that uses NLP techniques and operations (no LLM)
- The main goal of the chatbot is to recommend you recipes based on the ingredients and prep time you have.
- You can use this chatbot if you want to try new recipe ideas

# How to run
<h3> 1.) via Streamlit Cloud </h3>
Demo link (Streamlit): https://projectnlpantrychatbot-y5bk3vsp9rysvuzakjxgzp.streamlit.app/ 

<h3> 2.) Python (created with version 3.11.8) </h3>
Install the required libraries by the following command:

    pip install -r requirements.txt

Then run streamlit locally using this command:

    streamlit run streamlit_app.py

# Text pre-processing
The user’s messages were cleaned to ensure detection of important keywords, such as commands and ingredient names. This was done via the following steps:

1.)	Lower case conversion

2.)	Punctuation removal

3.)	Word Tokenization

4.)	Stop word removal

5.)	Lemmatization

# Dialogue Flow
The chatbot system has five main steps. The dialogue flow is shown by the following:

<h2> Step 1: Ingredients Collection Stage </h2>

This stage involves collecting the user’s list of ingredients. The user must add ingredients one by one. This stage was done by detecting two types of keywords:

<h3> 1.) Ingredients name </h3>

- This was achieved by applying Named Entity Recognition (NER) using Spacy. 
- However, the default spacy model (en_core_web_sm) cannot recognize food items or ingredient names.
- To address this limitation, a master list of ingredients from the ingredients.txt file was added to the Spacy model using the PhraseMatcher function.
- This allowed the system to tag matched tokens with the ‘Ingredients’ tag, allowing detection of ingredient names in the user’s input

```python
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr = "LOWER")
#Match all the ingredient names with the "INGREDIENT" tag
matcher.add("INGREDIENT", ingredients_doc)
```
<h3> 2.) Commands </h3>
The system detects the following commands: Add, Remove, Clear, Done/Finished

To recognize similar commands, the WordNet function from NLTK was used to obtain a list of synonymous words. The user’s tokens are then compared with this synonym list.

<h2> Step 2: Get available cook time  </h2>

- This stage involves asking the user how much available time they have to make a dish.
- The user's time input is converted into minutes. An example input is "1 hour and 30 minutes" which is extracted as 90 minutes


<h2> Step 3: Match Recipe Stage  </h2>

- This stage matches the five most relevant recipe based on the user’s ingredients list. 
- To achieve this, the dataset and the user’s ingredients list were converted into vectors using TF-IDF. 
- TF-IDF assigns numerical weights to each word in the document based on their frequency and importance.

```python
tfidf = TfidfVectorizer(lowercase = True, vocabulary = ingredients_list, min_df = 1)
df_vector = tfidf.fit_transform(ingredients_dataset)
```

- Following this, the TF-IDF vectors were compared using cosine similarity.
- Cosine similarity is an efficient method of measuring text similarity and is commonly used in recommendation systems.
- The similarity scores were then ranked, and the five highest similarity scoring recipes were given as the options to the user

```python
user_vector = df_tfidf.transform(user_ingredients) #Convert the user's input into vector
#Compare both vectors using cosine similarity
similarity = cosine_similarity(user_vector, df_vector).flatten()
#Sort the recipe based on cosine similarity score
sorted_recipe = np.argsort(similarity)[::-1] #sort by descending
```

<h2> Step 4: Selecting Recipe Stage  </h2>

During this stage the user must select one of the five given recipes. This can be done by using any of these three methods:

    1.)	Selecting by specifying the number (e.g. “I want recipe 1”)
    
    2.)	Selecting by numbered words or ordinal (e.g. “I want the first recipe”)
    
    3.)	Selecting by name of recipe (e.g. “I want the Thai Pineapple Chicken Curry recipe”)

- To achieve the third method of selecting a recipe by name, the BLEU score was used
- BLEU score is an efficient method of measuring word similarities between two strings.
- This can help match a recipe name even when the user provided only parts of the recipe’s name or the words at different order.
- For example, if one of the options is “Chicken and Spinach Wild Rice Soup”, the chatbot can recognize “Chicken and Spinach” or “Wild Soup” as inputs.

<h2> Step 5: Printing Recipe Information  </h2>

This final stage simply prints out the recipe selected by the user. This includes the full list of ingredients and the step-by-step instructions needed to complete the recipe. Additionally, the user is given the option to return to the recipe selection stage or the ingredients collection stage.

Example output:

<img width="940" height="716" alt="image" src="https://github.com/user-attachments/assets/7331fa97-21d1-4ed3-ada0-bb20cc968a32" />
