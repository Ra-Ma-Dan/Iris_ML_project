import joblib
import numpy as np
from flask import render_template, request, Flask 

the_model_path = "C:\\Users\\NCC\\Documents\\Ra Ma Dan\\ML Pro\\My_Models\\iris_predic_model.pkl"
iris_model = joblib.load(the_model_path)

static_path = "C:\\Users\\NCC\\Documents\\Ra Ma Dan\\Web pages\\Flasks\\Iris_Predictor\\static"
template_path = "C:\\Users\\NCC\\Documents\\Ra Ma Dan\\Web pages\\Flasks\\Iris_Predictor\\template"
app = Flask('__name__', template_folder=template_path, static_folder=static_path)

app.config['EXPLAIN_TEMPLATE_LOADING'] = True
@app.route('/', methods=['POST', 'GET'])
def index():
    
    final_prediction = None
    sep_length = None
    sep_width = None
    pet_length = None
    pet_width = None
    if request.method == 'POST':
        sep_length = request.form.get("sep_length")
        sep_width = request.form.get("sep_width")
        pet_length = request.form.get("pet_length")
        pet_width = request.form.get("pet_width")

        prediction_data = [sep_length, sep_width, pet_length, pet_width]
        prediction_data = np.array(prediction_data).reshape(1, -1)
        prediction_data = prediction_data.astype(float)
        

        flowers = ['setosa', 'versicolor', 'virginica']
        model_prediction = iris_model.predict(prediction_data)[0]
        print(model_prediction)
        final_prediction = flowers[model_prediction]

    return render_template("index.html", prediction=final_prediction, sep_len=sep_length, sep_wed=sep_width,
                            pet_len=pet_length, pet_wed=pet_width)


if  __name__ == '__main__':
    app.run(debug=True)



