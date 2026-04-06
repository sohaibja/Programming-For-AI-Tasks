from flask import Flask, render_template, request
import requests
from pyswip import Prolog

app = Flask(__name__)

NASA_API_KEY = "DEMO_KEY"

prolog = Prolog()
prolog.consult("logic.pl")

@app.route("/", methods=["GET", "POST"])
def home():
    data = None
    logic_result = None

    if request.method == "POST":
        topic = request.form["topic"].lower()

        # Call NASA API
        url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}"
        response = requests.get(url)

        if response.status_code == 200:
            nasa_data = response.json()

            data = {
                "title": nasa_data["title"],
                "image": nasa_data["url"],
                "description": nasa_data["explanation"]
            }

        # Query Prolog (AI reasoning)
        result = list(prolog.query(f"interesting({topic})"))
        if result:
            logic_result = "AI says this topic is interesting!"
        else:
            logic_result = "AI has no knowledge about this topic."

    return render_template("index.html",
                           data=data,
                           logic_result=logic_result)

if __name__ == "__main__":
    app.run(debug=True)