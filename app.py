#importing libraries
import pickle
from flask import Flask, render_template, request
import numpy as np

#global Variable

app = Flask("app")
loaded_Model = pickle.load(open('KNN Model.pkl', 'rb'))
#route
@app.route('/')
def home():
    return render_template ("form.html")

@app.route("/prediction",methods=['POST'])
def predict():
    #Pclass', 'Age', 'SibSp', 'Parch', 'Fare'
    Pclass = request.form["Pclass"]
    Age = request.form["Age"]
    SibSp = request.form["SibSp"]
    Parch = request.form["Parch"]
    Fare = request.form["Fare"]

    prediction = loaded_Model.predict([[Pclass, Age, SibSp, Parch, Fare]])
    probability = loaded_Model.predict_proba([[Pclass, Age, SibSp, Parch, Fare]])
    probability = np.round((np.max(probability) * 100), 2)
    output = ""
    probability = f"{probability}%"

    if prediction[0] == 0:
        output = "person is not died"
    else:
        output = "person is died"

    print(prediction, probability)     

    return render_template ("form.html", output_prediction=output)




#main functions
if __name__ == '__main__':
    app.run(debug=True)
