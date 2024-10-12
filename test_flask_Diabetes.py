import numpy as np
from flask import Flask,request,jsonify
import pickle


app=Flask(__name__)
model=pickle.load(open('DiabetesModel.save','rb'))
scaler=pickle.load(open('StandardScaler.save','rb'))
@app.route('/predict',methods=['POST'])
def predict():
    # Get the data from the POST request
    Glucose = request.values.get('Glucose')
    BloodPressure = request.values.get('BloodPressure')
    BMI = request.values.get('BMI')

    # Extract the features
    features = [Glucose, BloodPressure, BMI]
    features = np.array(features).reshape(1, -1)
    features=scaler.transform(features)

    # Make prediction
    prediction = model.predict(features)[0]

    # Return the result as JSON
    return {'prediction': str(prediction)}


if __name__=='__main__':
  app.run(debug=True)