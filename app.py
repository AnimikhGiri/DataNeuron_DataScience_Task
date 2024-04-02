from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the pre-trained model
# Make sure the model from Part A is accessible here, update the path if needed
model_path = './similarity_model'
model = SentenceTransformer(model_path)

@app.route('/similarity', methods=['GET', 'POST'])
def similarity():
    if request.method == 'POST':
        content = request.get_json()
        text1 = content['text1']
        text2 = content['text2']
    elif request.method == 'GET':
        text1 = request.args.get('text1')
        text2 = request.args.get('text2')


    # Generate embeddings and compute similarity
    with torch.no_grad():
        embedding1 = model.encode([text1], convert_to_tensor=True)
        embedding2 = model.encode([text2], convert_to_tensor=True)
        similarity_score = cosine_similarity(embedding1, embedding2)[0][0]

    return jsonify({'similarity score': float(similarity_score)})

if __name__ == '__main__':
    app.run(debug=True)
