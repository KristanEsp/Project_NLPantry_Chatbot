import streamlit as st
import dialogue

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

#Run the streamlit app
st.title("NLPantry")
#Create the dialogue class and store it in streamlit's session state
if 'dialogue' not in st.session_state:
    st.session_state['dialogue'] = dialogue.DialoguePolicy()
dialogue = st.session_state['dialogue']

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display the previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Display welcome message
if dialogue.current_stage == "welcome_stage":
    dialogue.print_welcome_message()

#Get user's response and process it
prompt = st.chat_input("Type Here.")
if prompt:
    if dialogue.current_stage == "ingredients_collection":
        dialogue.choose_ingredient_stage(prompt)
    if dialogue.current_stage == "choose_recipe_stage":
        dialogue.choose_recipe_stage(prompt)
    if dialogue.current_stage == "get_cook_time":
        dialogue.get_cook_time_stage(prompt)
    if dialogue.current_stage == "show_recipe_stage":
        dialogue.print_recipe_stage(prompt)
    if dialogue.current_stage == "end_stage":
        dialogue.end_stage(prompt)

    display_streamlit_chat(prompt, dialogue.chatbot_output)