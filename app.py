from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    values = list(data.values())
    arr = np.array([values])
    arr_scaled = scaler.transform(arr)
    result = model.predict(arr_scaled)
    return jsonify({"prediction": int(result[0])})

if __name__ == "__main__":
    app.run(debug=True)
