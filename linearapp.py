from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and data at startup
lr = joblib.load('gdp_linear_regression_model.pkl')
df = pd.read_csv('gdp_population_with_lags.csv')

features = ["Population", "GDP_lag_1", "GDP_lag_2", "GDP_lag_3", 
            "GDP_growth_1yr", "Population_growth_1yr", "GDP_per_capita"]

# Precompute lag features for all data
for lag in [1, 2, 3]:
    df[f"GDP_lag_{lag}"] = df.groupby("Country Name")["GDP"].shift(lag)

df["GDP_per_capita"] = df["GDP"] / df["Population"]
df["GDP_growth_1yr"] = df.groupby("Country Name")["GDP"].pct_change().fillna(0)
df["Population_growth_1yr"] = df.groupby("Country Name")["Population"].pct_change().fillna(0)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    country = data.get('country')
    year = data.get('year')
    
    # Get the country's most recent data
    country_data = df[df["Country Name"] == country].sort_values("Year")
    
    if country_data.empty:
        return jsonify({'error': f'Country "{country}" not found'}), 404
    
    # Get the last available row for this country
    last_row = country_data.iloc[-1].copy()
    
    # Predict iteratively from last known year to target year
    last_known_year = int(last_row["Year"])
    
    for y in range(last_known_year + 1, year + 1):
        X_input = last_row[features].values.reshape(1, -1)
        pred_gdp = lr.predict(X_input)[0]
        
        # Update features for next iteration
        last_row["GDP_lag_3"] = last_row["GDP_lag_2"]
        last_row["GDP_lag_2"] = last_row["GDP_lag_1"]
        last_row["GDP_lag_1"] = pred_gdp
        last_row["GDP_growth_1yr"] = (pred_gdp - last_row["GDP_lag_2"]) / last_row["GDP_lag_2"]
        last_row["Year"] = y
    
    return jsonify({
        'country': country,
        'year': year,
        'predicted_gdp': pred_gdp
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)