from flask import Flask, render_template,jsonify,request
import pandas as pd
import pickle
app = Flask(__name__)

@app.route('/')

def home():
    return render_template('homepage.html')


@app.route("/predict",methods = ["GET","POST"])

def predict():
    if request.method == "POST":
        make = request.form.get("make")
        model = request.form.get("model")
        year = request.form.get("year")
        engineCylinders = request.form.get("engineCylinders")
        engineFuelType = request.form.get("engineFuelType")
        engineHP = request.form.get("engineHP")
        da=pd.read_json("new.json")
        make_encode= da['Make_encode'][da['Make']==make].values[0]
        model_encode= da['Model_encode'][da['Model']==model].values[0]
        fuel_Type=da['Engine Fuel Type_encode'][da['Engine Fuel Type']==engineFuelType].values[0]
        
        print(make_encode,model_encode,fuel_Type)

        with open('model.pkl','rb') as mod:
            mlmodel = pickle.load(mod)

        predict = mlmodel.predict([[make_encode,model_encode,float(year),float(engineCylinders),float(engineHP)]])

        return render_template('predict.html',predicted_value=predict[0])

    else:
        return render_template('predict.html')
    
@app.route('/Submit')

def Submit():
    return render_template('Submit.html')
if __name__=='__main__':
    app.run(host='0.0.0.0')






