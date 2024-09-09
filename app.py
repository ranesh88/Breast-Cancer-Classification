from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


def predict(mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness):
    # Prepare features array
    features = np.array([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]])

    # scaling
    scaled_features = scaler.transform(features)

    # predict by model
    result = model.predict(scaled_features)

    return result[0]

# Routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        mean_radius = float(request.form['mean_radius'])
        mean_texture = float(request.form['mean_texture'])
        mean_perimeter =float(request.form['mean_perimeter'])
        mean_area = float(request.form['mean_area'])
        mean_smoothness = float(request.form['mean_smoothness'])


        prediction = predict(mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness)
        prediction_text = "The Patient has Breast Cancer" if prediction == 1 else "The Patient has no Breast Cancer"

        return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)