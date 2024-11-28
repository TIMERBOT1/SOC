from flask import Flask, request, jsonify
from langchain_core.load import loads
import json

app = Flask(__name__)

def generate_answer(question):
    with open("/path/to/your/chain.json", "r") as file:
        string_representation = file.read()

    chain = loads(string_representation)
    return json.loads(chain.invoke(question))

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    
    if 'question' not in data:
        return jsonify({'error': 'Вопрос не задан.'}), 400
    
    question = data['question']
    answer = generate_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)