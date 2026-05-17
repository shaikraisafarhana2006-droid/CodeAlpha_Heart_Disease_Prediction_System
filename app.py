from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# LOAD MODEL
model = pickle.load(open("models/heart_model.pkl", "rb"))

# LOAD SCALER
scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # SCALE INPUT
    input_data = scaler.transform(input_data)

    # PREDICT
    prediction = model.predict(input_data)

    # PROBABILITY
    if int(prediction[0]) == 0:
        probability = model.predict_proba(input_data)[0][0] * 100
    else:
        probability = model.predict_proba(input_data)[0][1] * 100

    print(prediction)
    print("Prediction Value:", prediction[0])

    if int(prediction[0]) == 0:
        result = "❤️ Heart Disease Detected"
    else:
        result = "💚 No Heart Disease"

    return render_template(
        "index.html",
        prediction_text=result,
        probability=round(probability, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)