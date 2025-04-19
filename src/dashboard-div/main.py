import dash
from dash import dcc, html, Input, Output
import requests

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Fraud Detection Model Prediction"),
    
    # Input fields for each feature
    dcc.Input(id='input-Time', type='number', placeholder='Time'),
    dcc.Input(id='input-V1', type='number', placeholder='V1'),
    dcc.Input(id='input-V2', type='number', placeholder='V2'),
    dcc.Input(id='input-V3', type='number', placeholder='V3'),
    dcc.Input(id='input-V4', type='number', placeholder='V4'),
    dcc.Input(id='input-V5', type='number', placeholder='V5'),
    dcc.Input(id='input-V6', type='number', placeholder='V6'),
    dcc.Input(id='input-V7', type='number', placeholder='V7'),
    dcc.Input(id='input-V8', type='number', placeholder='V8'),
    dcc.Input(id='input-V9', type='number', placeholder='V9'),
    dcc.Input(id='input-V10', type='number', placeholder='V10'),
    dcc.Input(id='input-V11', type='number', placeholder='V11'),
    dcc.Input(id='input-V12', type='number', placeholder='V12'),
    dcc.Input(id='input-V13', type='number', placeholder='V13'),
    dcc.Input(id='input-V14', type='number', placeholder='V14'),
    dcc.Input(id='input-V15', type='number', placeholder='V15'),
    dcc.Input(id='input-V16', type='number', placeholder='V16'),
    dcc.Input(id='input-V17', type='number', placeholder='V17'),
    dcc.Input(id='input-V18', type='number', placeholder='V18'),
    dcc.Input(id='input-V19', type='number', placeholder='V19'),
    dcc.Input(id='input-V20', type='number', placeholder='V20'),
    dcc.Input(id='input-V21', type='number', placeholder='V21'),
    dcc.Input(id='input-V22', type='number', placeholder='V22'),
    dcc.Input(id='input-V23', type='number', placeholder='V23'),
    dcc.Input(id='input-V24', type='number', placeholder='V24'),
    dcc.Input(id='input-V25', type='number', placeholder='V25'),
    dcc.Input(id='input-V26', type='number', placeholder='V26'),
    dcc.Input(id='input-V27', type='number', placeholder='V27'),
    dcc.Input(id='input-V28', type='number', placeholder='V28'),
    dcc.Input(id='input-Amount', type='number', placeholder='Amount'),
    
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output', style={'margin-top': '20px'})
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [Input(f'input-{col}', 'value') for col in ['Time', 'V1', 'V2', 'V3', 'V4', 
                                                  'V5', 'V6', 'V7', 'V8', 'V9', 
                                                  'V10', 'V11', 'V12', 'V13', 
                                                  'V14', 'V15', 'V16', 'V17', 
                                                  'V18', 'V19', 'V20', 'V21', 
                                                  'V22', 'V23', 'V24', 'V25', 
                                                  'V26', 'V27', 'V28', 'Amount']]
)
def update_output(n_clicks, *values):
    if n_clicks > 0:
        # Prepare the input for the API
        input_data = [{
            "Time": values[0],
            "V1": values[1],
            "V2": values[2],
            "V3": values[3],
            "V4": values[4],
            "V5": values[5],
            "V6": values[6],
            "V7": values[7],
            "V8": values[8],
            "V9": values[9],
            "V10": values[10],
            "V11": values[11],
            "V12": values[12],
            "V13": values[13],
            "V14": values[14],
            "V15": values[15],
            "V16": values[16],
            "V17": values[17],
            "V18": values[18],
            "V19": values[19],
            "V20": values[20],
            "V21": values[21],
            "V22": values[22],
            "V23": values[23],
            "V24": values[24],
            "V25": values[25],
            "V26": values[26],
            "V27": values[27],
            "V28": values[28],
            "Amount": values[29]
        }]
        
        # Call the prediction API
        response = requests.post('http://127.0.0.1:5000/predict', json=input_data)
        
        if response.status_code == 200:
            prediction = response.json().get('predictions', [])
            return f'Prediction: {prediction[0]}'
        else:
            return f'Error: {response.json().get("error", "Unknown error")}'
    return "Click the button to make a prediction."

if __name__ == '__main__':
    app.run(debug=True, port=8050)
