from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from joblib import load
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the model from the file
loaded_model = load('models/naive_bayes_model.joblib')
tfidf_vectorizer = load('models/tfidf_vectorizer.joblib')

# load tokenization
data_df = pd.read_excel('tokenization/Pahamify_Tokenization.xlsx')
positif_df = len(data_df[data_df['label'] == 'positif'])
negatif_df = len(data_df[data_df['label'] == 'negatif'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    # Get the input text from the request
    data = request.json
    tes_model = data['text']

    # Convert the text into feature vector using the same TF-IDF Vectorizer
    tfidf_vector_tes = tfidf_vectorizer.transform([tes_model])

    # Perform prediction using the loaded model
    prediksi = loaded_model.predict(tfidf_vector_tes)
    if prediksi[0] == 0:
        Hasil = 'Negatif'
    else :
        Hasil = 'Positif'

    # Return the prediction result
    return jsonify({
        'text': tes_model,
        'prediction': Hasil
    })

# http://localhost:5000/image/umum.png
@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('img', filename)

@app.route('/chart')
def chart():
    return jsonify({
        'positif' : positif_df,
        'negatif': negatif_df
    })

@app.route('/sample')
def sample_data():
    top_5_df = data_df.head(5)
    
    top_5_dict = top_5_df.to_dict(orient='records')
    
    # Return the top 10 rows as JSON
    return jsonify(top_5_dict)

if __name__ == '__main__':
    app.run(debug=True)
