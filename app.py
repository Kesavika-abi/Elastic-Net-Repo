from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained Elastic Net model
model = joblib.load("elastic_net_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        goal_amount = float(request.form["Goal_Amount"])
        campaign_length = int(request.form["Campaign_Length"])
        backers_first_48h = int(request.form["Backers_First_48H"])
        social_media_shares = int(request.form["Social_Media_Shares"])
        category_popularity = float(request.form["Category_Popularity"])
        video_views = int(request.form["Video_Views"])
        updates_posted = int(request.form["Updates_Posted"])

        # Prepare input for prediction
        features = np.array([[goal_amount, campaign_length, backers_first_48h,
                               social_media_shares, category_popularity,
                               video_views, updates_posted]])

        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
