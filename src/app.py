from flask import Flask, jsonify
from src.load_data import load_data
from src.preprocess import preprocess_data
from src.train_model import train_and_evaluate

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/train", methods=["POST"])
def train():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    _, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)

    return jsonify({"model_accuracy": accuracy})

if __name__ == "__main__":
    app.run(debug=True)
