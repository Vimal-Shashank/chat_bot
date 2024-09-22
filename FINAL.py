from flask import Flask, request, jsonify

import os
from educhain import qna_engine
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

app = Flask(__name__)

# Initialize the parser
parser = StrOutputParser()

# Securely load the API key


# Initialize the ChatGroq model
llmu = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.5,
    groq_api_key=""
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('user_input')

    if not user_input:
        return jsonify({"error": "Please provide a user input."}), 400

    if user_input.lower() == 'bye':
        return jsonify({"response": "Goodbye! Have a great day!"})

    try:
        human_message = HumanMessage(content=user_input)
        response = parser.invoke(llmu.invoke([human_message]))
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_mcqs_from_url', methods=['POST'])
def generate_mcqs_from_url():
    data = request.json
    url_input = data.get('url_input')
    learning_objective_url = data.get('learning_objective_url')

    if not url_input or not learning_objective_url:
        return jsonify({"error": "Please provide both the URL and the learning objective."}), 400

    try:
        url_mcqs = qna_engine.generate_mcqs_from_data(
            source=url_input,
            source_type="url",
            num=3,
            learning_objective=learning_objective_url,
            llm=llmu
        )
        mcqs = [{"question": mcq.question, "answer": mcq.answer, "explanation": mcq.explanation, "options": mcq.options} for mcq in url_mcqs.questions]
        return jsonify({"mcqs": mcqs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_mcqs_from_topic', methods=['POST'])
def generate_mcqs_from_topic():
    data = request.json
    topic_input = data.get('topic_input')
    learning_objective_topic = data.get('learning_objective_topic')
    difficulty_level = data.get('difficulty_level')

    if not topic_input or not learning_objective_topic or not difficulty_level:
        return jsonify({"error": "Please provide the topic, learning objective, and difficulty level."}), 400

    try:
        custom_template = """
        Generate {num} multiple-choice question (MCQ) based on the given topic and level.
        Provide the question, four answer options, and the correct answer.

        Topic: {topic}
        Learning Objective: {learning_objective}
        Difficulty Level: {difficulty_level}
        """

        result = qna_engine.generate_mcq(
            topic=topic_input,
            num=5,
            learning_objective=learning_objective_topic,
            difficulty_level=difficulty_level,
            prompt_template=custom_template,
            llm=llmu
        )
        mcqs = [{"question": mcq.question, "answer": mcq.answer, "explanation": mcq.explanation, "options": mcq.options} for mcq in result.questions]
        return jsonify({"mcqs": mcqs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
