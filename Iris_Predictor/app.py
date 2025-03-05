from flask import render_template, request, Flask 


template_path = "C:\\Users\\NCC\\Documents\\Ra Ma Dan\\Web pages\\Flasks\\Iris_Predictor\\template"
app = Flask('__name__', template_folder=template_path)

app.config['EXPLAIN_TEMPLATE_LOADING'] = True
@app.route("/")
def index():
    return render_template("index.html")


if  __name__ == '__main__':
    app.run(debug=True)