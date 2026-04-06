from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_KEY = "YOUR_API_KEY_HERE"

@app.route("/", methods=["GET", "POST"])
def home():
    weather_data = None
    suggestion = None

    if request.method == "POST":
        city = request.form["city"]

        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            weather_data = {
                "city": city,
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"]
            }

            #  Production Rule (Knowledge Representation)
            if "rain" in weather_data["description"].lower():
                suggestion = "Take an umbrella ☔"
            else:
                suggestion = "Weather is clear 🌤"

        else:
            suggestion = "City not found!"

    return render_template("index.html",
                           weather=weather_data,
                           suggestion=suggestion)


if __name__ == "__main__":
    app.run(debug=True)