from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)



# Load the trained model and scalers
model = joblib.load('Models/crop_prediction_model.pkl')
scaler = joblib.load('Models/scaler.pkl')
le_state = joblib.load('Models/label_encoder_state.pkl')
le_crop = joblib.load('Models/label_encoder_crop.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/croprecommended')
def cropform():
    return render_template('croprecommended.html')

@app.route('/cropresult', methods=['POST'])
def predict():
    data = request.form.to_dict()
    try:
        data['N_SOIL'] = float(data['N_SOIL'])
        data['P_SOIL'] = float(data['P_SOIL'])
        data['K_SOIL'] = float(data['K_SOIL'])
        data['TEMPERATURE'] = float(data['TEMPERATURE'])
        data['HUMIDITY'] = float(data['HUMIDITY'])
        data['ph'] = float(data['ph'])
        data['RAINFALL'] = float(data['RAINFALL'])

        # Encode the state, with handling for unseen labels
        if data['STATE'] in le_state.classes_:
            data['STATE'] = le_state.transform([data['STATE']])[0]
        else:
            return jsonify({'error': f"Unseen label: {data['STATE']}"}), 400

        # Convert the input into a DataFrame
        df = pd.DataFrame([data])

        # Scale the features
        df_scaled = scaler.transform(df)

        # Predict the crop
        prediction = model.predict(df_scaled)
        predicted_crop = le_crop.inverse_transform(prediction)

        return render_template('result.html', crop=predicted_crop[0])

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
