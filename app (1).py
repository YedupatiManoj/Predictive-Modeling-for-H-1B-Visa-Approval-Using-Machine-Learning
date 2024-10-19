from flask import Flask, request, render_template
import pandas as pd
import pickle  # Importing pickle for saving/loading models

app = Flask(__name__)

# Load the model
model_file_path = 'C:/Visa_Approval_Prediction/model/visraf.pkl'  # Update this path


# Open the model file
with open(model_file_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('visaapproval.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    full_time_position = request.form['full_time_position']
    prevailing_wage = float(request.form['prevailing_wage'])
    year = int(request.form['year'])
    soc_name = request.form['soc_name']

    # Preprocessing the input data
    full_time_position = 1 if full_time_position == 'Y' else 0
    soc_n = 0 if soc_name.lower() == 'it' else 1  # Adjust according to your mapping

    # Prepare the feature array for prediction
    input_data = pd.DataFrame([[full_time_position, prevailing_wage, year, soc_n]],
                              columns=["FULL_TIME_POSITION", "PREVAILING_WAGE", "YEAR", "SOC_N"])

    # Make prediction
    prediction = loaded_model.predict(input_data)
    prediction_result = 'Approved' if prediction[0] == 1 else 'Denied' if prediction[0] == 2 else 'Unknown'

    return render_template('resultVA.html', prediction_text=f'The visa is {prediction_result}.')

if __name__ == "__main__":
    app.run(debug=True)
