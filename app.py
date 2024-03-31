from flask import Flask,request,render_template
import numpy
import pandas
import sklearn
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorus'])
    K = int(request.form['Potassium'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    Ph = float(request.form['Ph'])
    Rainfall = float(request.form['Rainfall'])
    feature_list = [N, P, K, Temperature, Humidity, Ph, Rainfall]
    single_pred = numpy.array(feature_list).reshape(1, -1)
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    # scaled_features = ms.transform(single_pred)
    # # final_features = sc.transform(scaled_features)
    prediction = model.predict(single_pred)
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated within the given conditions".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)


if __name__ == "__main__":
    app.run(debug=True)
