from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        study_hours = float(data["study_hours"])
        attendance = float(data["attendance"])
        assignment_score = float(data["assignment_score"])
        previous_score = float(data["previous_score"])
        sleep_hours = float(data["sleep_hours"])

        # Prediction
        prediction = model.predict([[
            study_hours,
            attendance,
            assignment_score,
            previous_score,
            sleep_hours
        ]])

        predicted_score = round(float(prediction[0]), 2)

        # Keep score between 0 and 100
        predicted_score = max(0, min(100, predicted_score))

        # Performance category
        if predicted_score >= 90:
            performance = "Excellent"
        elif predicted_score >= 75:
            performance = "Good"
        elif predicted_score >= 60:
            performance = "Average"
        else:
            performance = "Needs Improvement"

        return jsonify({
            "predicted_score": predicted_score,
            "performance": performance
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)