from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('bitcoin_price_model.pkl')

# Function to create a bar chart
def create_bar_chart(open_price, high, low, volume, market_cap, prediction):
    # Set up the bar chart data
    labels = ['Open', 'High', 'Low', 'Volume', 'Market Cap', 'Predicted Close']
    values = [open_price, high, low, volume, market_cap, prediction]
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
    # plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Bitcoin Price Prediction - Feature Analysis')
    
    # Save the chart to a static file
    chart_path = 'static/prediction_bar_chart.png'
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    start = float(request.form['start'])
    end = float(request.form['end'])
    open_price = float(request.form['open'])
    high = float(request.form['high'])
    low = float(request.form['low'])
    volume = float(request.form['volume'])
    market_cap = float(request.form['market_cap'])
    
    # Prediction
    features = np.array([[open_price, high, low, volume, market_cap]])
    prediction = model.predict(features)[0]
    
    # Generate the bar chart
    chart_path = create_bar_chart(open_price, high, low, volume, market_cap, prediction)
    
    return render_template('index.html', prediction=f"Predicted Close Price: ${prediction:.2f}", chart_path=chart_path)

if __name__ == '__main__':
    app.run(debug=True)
