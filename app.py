from flask import Flask, request, jsonify
import yfinance as yf
from fbprophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
# For LSTM, you'd typically use TensorFlow or PyTorch, but that's a bit more involved.

app = Flask(__name__)

@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    stock_name = request.json['stockName']
    data = yf.download(stock_name, start="2020-01-01", end="2023-01-01")
    return jsonify(data.to_dict())

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.json['stockName']
    model_type = request.json['modelType']
    data = yf.download(stock_name, start="2020-01-01", end="2023-01-01")
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']

    if model_type == "ARIMA":
        model = ARIMA(df['y'], order=(5,1,0))
        model_fit = model.fit(disp=0)
        predictions = model_fit.forecast(steps=30)[0]
    elif model_type == "Prophet":
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        predictions = forecast['yhat'][-30:].tolist()
    # LSTM would go here, but it's quite involved and would require a lot more setup.

    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)


