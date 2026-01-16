from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
import numpy as np
from src.preprocess import preprocess_data
from src.train_model import train_and_evaluate

app = Flask(__name__)

# Health check endpoint (TC5)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# Simple prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from request
        data = request.json["features"]
        input_data = np.array(data).reshape(1, -1)

        # Train model and get accuracy (baseline)
        accuracy = train_and_evaluate()

        return jsonify({
            "message": "Prediction endpoint working",
            "model_accuracy": accuracy
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

