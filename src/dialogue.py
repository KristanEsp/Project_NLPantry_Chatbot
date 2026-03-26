import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt_tab')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.translate.bleu_score import sentence_bleu
import spacy
import streamlit as st
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Import functions from other notebook
from nlp_processing import text_preprocessing, nlp_processing, load_ingredients_list, tag_ingredients, identify_ingredients, perform_tfidf, perform_cosine_similarity


# In[13]:




def display_streamlit_chat(prompt, response):
    #Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    #Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    #Display chat bot response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    #Add chat bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


def check_valid_ingredients(self, ingredient, ingredients_list):
    #Check if any valid ingredients were added
    if ingredient is None:
        self.chatbot_output = "Ingredient was not recognized. Please try again"
        return False
    #Check if ingredient is already on the list
    elif ingredient in ingredients_list:
        self.chatbot_output = f'You have already added: {ingredient}'
        return False
        #print("You have already added:", ingredient)
    else:
        return True


# In[14]:


#Lemmatized verbs or commands e.g.:finished->finish
def lemmatized_verbs(token):
    wn = nltk.WordNetLemmatizer()
    text_lemmatized = []
    for word,tag in pos_tag(token):
        #Check if the token/word has a POS verb tag
        if tag[0] == "V" and word != "done":
             text_lemmatized.append(wn.lemmatize(word, tag[0].lower())) #Use the "v" tag to indicate that its a verb
        else:
            text_lemmatized.append(wn.lemmatize(word))
    return text_lemmatized


# In[15]:


#Check if a token is synonymous/matches a specific command such as "finished" or "done"
def check_similar_commands(tokens, commands):
    # Using Word2 net to find synonyms of the word "finish"
    synonyms = []
    for command in commands:
        for synonym in wn.synsets(command):
            for lemma in synonym.lemma_names():
                synonyms.append(lemma)

    #Ensure that the commands are lemmatized
    tokens = lemmatized_verbs(tokens)
    #Check if the user's token matches that of any word in the synonym list
    for token in tokens:
        if token in synonyms:
            return commands[0]
        else:
            return


# In[16]:


def convert_words2number(token):
    words_to_nums = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5"
    }
    return words_to_nums.get(token)


# In[28]:


#using Bleu score to match the recipe name given by the user to one of the top 5 recipe list
def get_bleu_score(self, tokens):
    tokens = [tokens]
    #Get the index positions of the top 5 recipes
    index = self.top5_recipe.index
    chosen_index = 0
    #Calculate the bleu score for each recipe list
    bleu_score = 0
    for i, recipe in enumerate(self.top5_recipe):
        #Perform text cleaning on recipe name
        recipe = text_preprocessing(recipe)
        recipe = word_tokenize(recipe)
        #Calculate the bleu score
        score = sentence_bleu(tokens, recipe)
        if score > bleu_score:
            bleu_score = score
            chosen_index = index[i]
            
    #If no matches found return
    if bleu_score == 0:
        return
    #Otherwise return the recipe's index with the highest bleu score
    else:
        return chosen_index


# In[29]:


#
def check_recipe_response(tokens, self):
    #First, check if user used numbers/digit to pick a recipe from the list
    for token in tokens:
        #Check for worded numbers and convert them to digits
        number_choice = convert_words2number(token)
        if number_choice is not None:
            token = number_choice
        #Check for numbers/digits and check if this number is within range of 1-5
        if token.isnumeric() and int(token) in range(1,6):
            index = self.top5_recipe.index
            token = int(token) #Convert to int
            chosen_index = index[token - 1]
            #Return the dataframe index of the user's choice
            return chosen_index
    #Second, if user entered a name instead find a matchin one based on Bleu score   
    chosen_index = get_bleu_score(self, tokens)
    return chosen_index


# In[167]:


def print_info(self, info, pattern):
    #Remove punctuations except for comma
    punctuations_with_comma = r'[\[\]]'
    info_list = re.sub(punctuations_with_comma, "", info) #Clean unwanted punctuations
    info_list = re.split(pattern, info_list)
    i = 0
    #Print the info in a numbered list
    for info in info_list:
        info = re.sub(r"['\"]", "", info) #Cleanup any remaining quotation marks
        #Check if info is empty if so skip
        if len(info) == 0:
            continue
        self.chatbot_output += f"{i+1}.) {info}  \n"
        ############# print(f"{i+1}.) {info}")
        i += 1


# In[168]:


#Dialogue Policy
class DialoguePolicy:
    def __init__(self):
        self.ingredients_list = []
        self.top5_recipe = []
        self.chosen_recipe_index = 0
        self.current_stage = "welcome_stage"
        self.user_input = ""
        self.chatbot_output = ""
        #Import the food dataset
        self.df_food = pd.read_csv(os.path.join(BASE_DIR, "dataset", "Recipe_Dataset.csv"))

    def print_welcome_message(self):
        self.chatbot_output = f'I am NLPantry a chatbot that can recommend you some recipes by giving me a list of ingredients.  \n'
        self.chatbot_output += f'• Start by adding any ingredients you have, one by one.  \n'
        self.chatbot_output += f'• You can also "remove" ingredients from the list or "clear" the entire list  \n'
        self.chatbot_output += f'• Let me know once you are done collecting your ingredients list  \n'
        self.current_stage = "ingredients_collection"
        #Display chat bot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(self.chatbot_output)
        #Add chat bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": self.chatbot_output})
        st.rerun()

    ####################3 main stages: ingredients collection, recipe choosing, recipe output
    ############### Step 1.) Ingredients Collection Stage
    def choose_ingredient_stage(self, input):
        status = ""
        #Get user response
        self.user_input = input
        #Apply text cleaning to response
        response = text_preprocessing(self.user_input)
        response,tokens = nlp_processing(response)

        #Check if the user's response contains the finished or done command
        commands = ["done", "finished"]
        status = check_similar_commands(tokens, commands)

        #Move on to the choosing recipe stage after finishing adding ingredients
        if status == "done":
            #Check if ingredients list is not empty before moving to next stage
            if len(self.ingredients_list) != 0:
                self.ingredients_list = [", ".join(self.ingredients_list)]
                self.current_stage = "choose_recipe_stage"
                self.match_recipe()
            else:
                self.chatbot_output = "Your ingredients list is empty. You must add some ingredients before proceeding"
                return
        
        #Option to clear the list
        commands = ["clear"]
        status = check_similar_commands(tokens, commands)
        if status == "clear":
            #Clear the ingredients list
            self.ingredients_list = []
            self.chatbot_output = "Your ingredients list has been cleared"
            return

        #Detect ingredients from text
        all_ingredients = load_ingredients_list()
        matcher, nlp = tag_ingredients(all_ingredients)
        ingredient = identify_ingredients(response, matcher, nlp)

        #Remove ingredient with remove command
        commands = ["remove", "exclude"]
        status = check_similar_commands(tokens, commands)
        self.chatbot_output = f'Removed {status} from the list  \n'
        if status == "remove":
            try:
                self.ingredients_list.remove(ingredient)
                self.chatbot_output = f'Removed {ingredient} from the list  \n'
                #Print the ingredients in a numbered list
                i = 0
                for ingredients in self.ingredients_list:
                    self.chatbot_output += f"{i+1}.) {ingredients}  \n"
                    i += 1
                return
            
            except ValueError:
                self.chatbot_output = f'The {ingredient} was not found in the list  \n'
                return
            
        #Add ingredients if valid
        is_valid_ingredient = check_valid_ingredients(self, ingredient, self.ingredients_list)
        if not is_valid_ingredient:
            return
        else:
            #Append valid ingredients to list
            self.ingredients_list.append(ingredient)
            #Print out the ingredients list
            self.chatbot_output = f'Your current Ingredients List:  \n'
            #Print the info in a numbered list
            i = 0
            for ingredients in self.ingredients_list:
                self.chatbot_output += f"{i+1}.) {ingredients}  \n"
                i += 1
        

        
    ############### Step 2.) Choosing Recipe Stage
    #Match Recipe via cosine similarity
    def match_recipe(self):
        #Process the ingredients list and the dataset - tfidf
        df_tfidf, df_vector = perform_tfidf(self.df_food["ingredients"])
        #Apply Cosine similarity to get top 5 matching recipes
        self.top5_recipe = perform_cosine_similarity(df_tfidf, df_vector, self.ingredients_list)
        self.top5_recipe = self.df_food["name"][self.top5_recipe]

        #Show the user the recommended recipe list
        self.chatbot_output = "Here are some matching recipes I have found for you:  \n"
        for i,recipe in enumerate(self.top5_recipe):
            self.chatbot_output  += f'{i + 1}.) {recipe}  \n'
        self.chatbot_output += "  \n Which recipe would you like to see?"
        display_streamlit_chat(self.user_input, self.chatbot_output)
        st.rerun()
        
    def choose_recipe_stage(self, input):
        #Ask user to pick their preferred recipe from list
        #Get user response
        #response = input("Which recipe would you like to see?:")

        #Clean the response
        response_cleaned = text_preprocessing(input)
        response_cleaned = word_tokenize(response_cleaned)
        self.chosen_recipe_index = check_recipe_response(response_cleaned, self)
        #Check if the response is valid
        if self.chosen_recipe_index is not None:
            #Move on to step 3: Print Chosen Recipe
            self.current_stage = "show_recipe_stage"
            return
        else:
            #print("I have not detected a matching recipe. Please try again")
            self.chatbot_output = "I have not detected a matching recipe. Please try again"

    ############### Step 3.) Print chosen recipe
    def print_recipe_stage(self, input):
        self.user_input = input
        self.chatbot_output = ""
        #Print the recipe name
        self.chatbot_output  += f'Here is the recipe for {self.df_food["name"].iloc[self.chosen_recipe_index]}:  \n'
        ###############################print(f'Here is the recipe for {self.df_food["name"].iloc[self.chosen_recipe_index]}:')
        #Print the ingredients list
        ################################print("Ingredients: ")
        self.chatbot_output  += f'  \n Ingredients:  \n'
        split_pattern = '","'
        print_info(self, self.df_food["ingredients_raw_str"].iloc[self.chosen_recipe_index], split_pattern)
        #Print the Step Instructions
        self.chatbot_output  += f'  \n Instructions:  \n'
        ###############################print("Instructions: ")
        split_pattern = "', '"
        print_info(self, self.df_food["steps"].iloc[self.chosen_recipe_index], split_pattern)


        #Ask user to either pick another recipe or redo their ingredients list 
        self.chatbot_output += f' \n Would you like to edit your "ingredients" list or pick another "recipe" \n' 
        self.current_stage = "end_stage"
        display_streamlit_chat(self.user_input, self.chatbot_output)
        st.rerun()
        #response = input("Would you like to edit your 'ingredients' list or pick another 'recipe'")

    
        ############### Step 4.) Option to pick another recipe or redo their ingredients list
    def end_stage(self, input):
        self.user_input = input
        #Clean the response 
        response_cleaned = text_preprocessing(self.user_input) 
        response_cleaned, tokens = nlp_processing(response_cleaned) 
        #Check which option (recipe or ingredients) the user picked 
        if "recipe" in response_cleaned: 
            #Loop back to recipe matching stage 
            self.current_stage = "choose_recipe_stage" 
            self.match_recipe() 
        elif "ingredient" in response_cleaned: 
            #Loop back to the ingredients collection stage 
            self.current_stage = "ingredients_collection" 
            #Print out the ingredients list 
            self.chatbot_output = f'Your current Ingredients List:  \n'
            self.ingredients_list = self.ingredients_list[0].split(', ')
            print(self.ingredients_list)
            i = 0
            for ingredients in self.ingredients_list: 
                self.chatbot_output += f"{i+1}.) {ingredients}  \n" 
                i += 1
            display_streamlit_chat(self.user_input, self.chatbot_output)
            st.rerun() 
        else: 
            self.chatbot_output = f'Please pick either "ingredients" to edit your ingredients list or "recipe" to choose another one \n'