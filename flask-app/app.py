from flask import Flask,render_template,request
import mlflow
from preprocessing_utility1 import normalize_text
import pickle
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/sahilshore/mlops-mini-project.mlflow')
dagshub.init(repo_owner='sahilshore', repo_name='mlops-mini-project', mlflow=True)

app = Flask(__name__)


#load the model
model_name = "my_model"
model_version = 3
 
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # clean and preprocess the input text
    normalized_text = normalize_text(text)

    # bow 
    vectorized_text = vectorizer.transform([normalized_text])
    # make a prediction
    result = model.predict(vectorized_text)
    return render_template('index.html', result=result)
app.run(debug=True)