from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    funding = float(request.form["funding"])
    employees = int(request.form["employees"])
    experience = float(request.form["experience"])
    company_age = float(request.form["company_age"])
    revenue = float(request.form["revenue"])
    expenses = float(request.form["expenses"])
    market_size = float(request.form["market_size"])
    customer_growth = float(request.form["customer_growth"])

    input_data = pd.DataFrame([[
        funding,
        employees,
        experience,
        company_age,
        revenue,
        expenses,
        market_size,
        customer_growth
    ]], columns=[
        "Funding",
        "Employees",
        "FounderExperience",
        "CompanyAge",
        "Revenue",
        "Expenses",
        "MarketSize",
        "CustomerGrowth"
    ])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    if prediction == 1:
        result = "Success"
    else:
        result = "Failure"

    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)