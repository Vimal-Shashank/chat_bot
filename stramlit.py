import streamlit as st
import os
from educhain import qna_engine
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

# Initialize the parser
parser = StrOutputParser()

# Securely load the API key
groq_api_key = os.getenv("OPENAI_API_KEY")
if not groq_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize the ChatGroq model
llmu = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.5,
    groq_api_key=groq_api_key
)

# Streamlit UI
st.title("Educational Content Generator")

# Section for user queries
st.write("Ask your questions below:")

# Input from the user
user_input = st.text_input("Write your query:", key="user_query")

if st.button("Submit", key="submit_query"):
    if user_input.lower() == 'bye':
        st.write("Goodbye! Have a great day!")
    else:
        # Display the user input
        st.write(f"**You:** {user_input}")
        
        # Create a HumanMessage object
        human_message = HumanMessage(content=user_input)
        
        # Show a loading spinner while processing
        with st.spinner("Processing..."):
            try:
                # Invoke the parser with the message
                response = parser.invoke(llmu.invoke([human_message]))
                # Display the response
                st.write(f"**Chatbot:** {response}")
            except Exception as e:
                st.error("An error occurred while processing your request.")
                st.write(f"Debug Info: {e}")
                # Optionally log the error for debugging
                st.write(f"Error details: {str(e)}")

# Section for selecting input type
input_type = st.selectbox("Input Type", ["Generate MCQs from URL", "Generate MCQs from Topic"], key="input_type")

if input_type == "Generate MCQs from URL":
    st.header("Generate MCQs from URL")
    url_input = st.text_input("Enter the URL:", key="url_input")
    learning_objective_url = st.text_input("Enter the learning objective:", key="learning_objective_url")

    if st.button("Generate MCQs from URL"):
        if not url_input or not learning_objective_url:
            st.error("Please provide both the URL and the learning objective.")
        else:
            with st.spinner("Generating MCQs..."):
                try:
                    # Generate MCQs from URL
                    url_mcqs = qna_engine.generate_mcqs_from_data(
                        source=url_input,
                        source_type="url",
                        num=3,
                        learning_objective=learning_objective_url,
                        llm=llmu
                    )
                    # Display the MCQs
                    st.write("**Generated MCQs from URL:**")
                    for mcq in url_mcqs.questions:
                        st.write(f"**Question:** {mcq.question}")
                        st.write(f"**Answer:** {mcq.answer}")
                        st.write(f"**Explanation:** {mcq.explanation}")
                        st.write("**Options:**")
                        for option in mcq.options:
                            st.write(f"- {option}")
                except Exception as e:
                    st.error("An error occurred while generating MCQs.")
                    st.write(f"Debug Info: {e}")
                    # Optionally log the error for debugging
                    st.write(f"Error details: {str(e)}")

elif input_type == "Generate MCQs from Topic":
    st.header("Generate MCQs from Topic")
    topic_input = st.text_input("Enter the topic:", key="topic_input")
    learning_objective_topic = st.text_input("Enter the learning objective:", key="learning_objective_topic")
    difficulty_level = st.selectbox("Select the difficulty level:", ["Easy", "Medium", "Hard"], key="difficulty_level")

    if st.button("Generate MCQs from Topic"):
        if not topic_input or not learning_objective_topic or not difficulty_level:
            st.error("Please provide the topic, learning objective, and difficulty level.")
        else:
            with st.spinner("Generating MCQs..."):
                try:
                    # Define the custom template
                    custom_template = """
                    Generate {num} multiple-choice question (MCQ) based on the given topic and level.
                    Provide the question, four answer options, and the correct answer.

                    Topic: {topic}
                    Learning Objective: {learning_objective}
                    Difficulty Level: {difficulty_level}
                    """

                    # Generate MCQs from topic
                    result = qna_engine.generate_mcq(
                        topic=topic_input,
                        num=5,
                        learning_objective=learning_objective_topic,
                        difficulty_level=difficulty_level,
                        prompt_template=custom_template,
                        llm=llmu
                    )
                    # Display the MCQs
                    st.write("**Generated MCQs from Topic:**")
                    for mcq in result.questions:
                        st.write(f"**Question:** {mcq.question}")
                        st.write(f"**Answer:** {mcq.answer}")
                        st.write(f"**Explanation:** {mcq.explanation}")
                        st.write("**Options:**")
                        for option in mcq.options:
                            st.write(f"- {option}")
                except Exception as e:
                    st.error("An error occurred while generating MCQs.")
                    st.write(f"Debug Info: {e}")
                    # Optionally log the error for debugging
                    st.write(f"Error details: {str(e)}")
