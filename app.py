from flask import Flask, render_template, request
from search import load_index, search as run_search

app = Flask(__name__)
idx = load_index()


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    if request.method == "POST":
        query = request.form.get("q", "").strip()
        if query:
            results = run_search(query, idx, top_k=5)
    return render_template("index.html", query=query, results=results)


if __name__ == "__main__":
    app.run(debug=True)
