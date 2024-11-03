from flask import Flask, request, render_template
import pickle, os

def validasi_inputan(form_data):
    errors = {}

    if not form_data.get("Close_1"):
        errors["Close_1"] = "Close_1 tidak boleh kosong."
    else:
        try:
            Close_1 = float(form_data.get("Close_1"))
        except ValueError:
            errors["Close_1"] = "Close_1 harus berupa angka."

    if not form_data.get("Close_2"):
        errors["Close_2"] = "Close_2 tidak boleh kosong."
    else:
        try:
            Close_2 = float(form_data.get("Close_2"))
        except ValueError:
            errors["Close_2"] = "Close_2 harus berupa angka."

    if not form_data.get("Close_3"):
        errors["Close_3"] = "Close_3 tidak boleh kosong."
    else:
        try:
            Close_3 = float(form_data.get("Close_3"))
        except ValueError:
            errors["Close_3"] = "Close_3 harus berupa angka."

    return errors

def validate_data(record):
    errors = {}
    if record["Close_1"] < 0 or record["Close_1"] > 40000:
        errors["Close_1"] = "Close_1 harus diantara 0.0 dan 1.0"

    if record["Close_2"] < 0 or record["Close_2"] > 40000:
        errors["Close_2"] = "Close_2 harus diantara 0.0 dan 1.0"

    if record["Close_3"] < 0 or record["Close_3"] > 40000:
        errors["Close_3"] = "Close_3 harus diantara 0.0 dan 1.0"

    return errors

# Load models
linear_model_load = pickle.load(open('best_bagging_model.sav', 'rb'))
scaler_load = pickle.load(open('scaler.sav', 'rb'))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
app.config["DEBUG"] = True

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = {}
    if request.method == "POST":
        # Validasi inpClose_n tidak boleh kosong
        errors = validasi_inputan(request.form)
        if not errors:
            record = {
                "Close_1": float(request.form.get("Close_1")),
                "Close_2": float(request.form.get("Close_2")),
                "Close_3": float(request.form.get("Close_3")),
            }

            errors = validate_data(record)
            if not errors:
                # Data input untuk prediksi
                input_data = [
                    record["Close_1"],
                    record["Close_2"],
                    record["Close_3"],
                ]

                # Normalisasi input data
                input_data_normalized = scaler_load.transform([input_data])

                # Membuat prediksi dari model
                predicted_value_normalized = linear_model_load.predict(input_data_normalized)

                # Menyesuaikan bentuk data untuk inverse_transform
                predicted_value_normalized_full = [[predicted_value_normalized[0], 0, 0]]
                predicted_value_full = scaler_load.inverse_transform(predicted_value_normalized_full)

                # Mengambil elemen prediksi pertama sebagai hasil akhir
                prediction = float(predicted_value_full[0][0])

    return render_template('index.html', prediction=prediction, errors=errors, record=request.form)

if __name__ == "__main__":
    app.run(debug=True)

