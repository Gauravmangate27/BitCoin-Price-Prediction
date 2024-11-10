from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained model
with open('modelv2.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define the main route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    input_graph = None
    output_graph = None

    if request.method == 'POST':
        try:
            # Get data from form
            open_price = float(request.form['open_price'])
            high_price = float(request.form['high_price'])
            low_price = float(request.form['low_price'])
            volume = float(request.form['volume'])
            daily_return = float(request.form['daily_return'])

            # Prepare the input data for prediction
            input_data = np.array([[open_price, high_price, low_price, volume, daily_return]])
            
            # Make prediction
            prediction = loaded_model.predict(input_data)[0]
            probabilities = loaded_model.predict_proba(input_data)[0]  # Get probabilities for each class

            # Interpret the prediction result
            result_text = "Up 📈" if prediction == 1 else "Down 📉"
            flash(f"The model predicts the price movement will be: {result_text}", "success")

            # Generate input feature and output probability graphs
            input_graph = generate_input_graph(open_price, high_price, low_price, volume, daily_return)
            output_graph = generate_output_graph(probabilities)

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for('index'))

    return render_template('index.html', input_graph=input_graph, output_graph=output_graph)

def generate_input_graph(open_price, high_price, low_price, volume, daily_return):
    """Generate a bar chart for input feature values and return as a base64 string."""
    plt.figure()
    x_labels = ['Open Price', 'High Price', 'Low Price', 'Volume', 'Daily Return']
    values = [open_price, high_price, low_price, volume, daily_return]
    plt.bar(x_labels, values, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Input Feature Values')
    
    # Save graph to file-like object
    input_img = io.BytesIO()
    plt.savefig(input_img, format='png')
    input_img.seek(0)
    plt.close()

    # Encode image as base64 string
    input_graph = base64.b64encode(input_img.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{input_graph}"

def generate_output_graph(probabilities):
    """Generate a bar chart for prediction probabilities and return as a base64 string."""
    plt.figure()
    labels = ['Down', 'Up']
    plt.bar(labels, probabilities, color=['salmon', 'lightgreen'])
    plt.xlabel('Price Movement')
    plt.ylabel('Probability')
    plt.title('Prediction Probability')
    plt.ylim(0, 1)

    # Save graph to file-like object
    output_img = io.BytesIO()
    plt.savefig(output_img, format='png')
    output_img.seek(0)
    plt.close()

    # Encode image as base64 string
    output_graph = base64.b64encode(output_img.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{output_graph}"

if __name__ == '__main__':
    app.run(debug=True)
