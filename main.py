import streamlit as st
from educhain import qna_engine
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
# Initialize the parser
parser = StrOutputParser()

# Initialize the ChatGroq model
llmu = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.5,
    groq_api_key=" your key"
)

# Streamlit UI
st.title("Chatbot Interface")
st.write("Ask your questions below:")

# Input from the user
user_input = st.text_input("Write your query:")

if st.button("Submit"):
    if user_input.lower() == 'bye':
        st.write("Goodbye! Have a great day!")
    else:
        # Display the user input
        st.write(f"**You:** {user_input}")
        
        # Create a HumanMessage object
        human_message = HumanMessage(content=user_input)
        
        # Invoke the parser with the message
        response = parser.invoke(llmu.invoke([human_message]))
        
        # Display the response
        st.write(f"**Chatbot:** {response}")
