# Run this in your notebook or script
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from tqdm import tqdm
import joblib

# --- SETTINGS ---
GDP_CSV = "gdp_long_1980_2022.csv"           # path to cleaned GDP long CSV
POP_CSV = "population_long_1980_2022.csv"     # path to cleaned population long CSV
PREDICT_TO = 2030
TRAIN_END_YEAR = 2022   # last year of observed GDP in your files
MIN_YEARS_FOR_POP_FORECAST = 5

# --- 1) Load data ---
gdp = pd.read_csv(GDP_CSV)
pop = pd.read_csv(POP_CSV)

# unify column names if necessary
gdp.columns = [c.strip() for c in gdp.columns]
pop.columns = [c.strip() for c in pop.columns]

# keep only relevant years (safety)
gdp = gdp[gdp['Year'] <= TRAIN_END_YEAR]
pop = pop[pop['Year'] <= TRAIN_END_YEAR]  # we will forecast pop beyond TRAIN_END_YEAR

# --- 2) Merge GDP + Population on country + year ---
df = pd.merge(
    gdp[['Country Name','Country Code','Year','GDP']],
    pop[['Country Name','Country Code','Year','Population']],
    on=['Country Name','Country Code','Year'],
    how='inner'
)

# drop rows with missing GDP or Population just in case
df = df.dropna(subset=['GDP','Population']).copy()

# --- 3) Basic feature engineering ---
df['Year_float'] = df['Year'].astype(int)
# log transform GDP and Population
df['log_GDP'] = np.log1p(df['GDP'])
df['log_Pop'] = np.log1p(df['Population'])

# sort
df = df.sort_values(['Country Name','Year']).reset_index(drop=True)

# create lag features for log_GDP (1,2,3 years)
for lag in [1,2,3]:
    df[f'lag_{lag}'] = df.groupby('Country Name')['log_GDP'].shift(lag)

# rolling mean of last 3 years for log_GDP
df['rolling_3'] = df.groupby('Country Name')['log_GDP'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

# drop rows with NaN lag_1 (we'll keep rows where at least lag_1 exists for training)
df_model = df.dropna(subset=['lag_1']).copy()

# --- 4) Encode country (pooled model) ---
le = LabelEncoder()
df_model['country_id'] = le.fit_transform(df_model['Country Code'])

# features and target
FEATURES = ['Year_float','log_Pop','lag_1','lag_2','lag_3','rolling_3','country_id']
# ensure missing lag_2/3 are filled (they can be NaN for early years) - fill with lag_1
df_model['lag_2'] = df_model['lag_2'].fillna(df_model['lag_1'])
df_model['lag_3'] = df_model['lag_3'].fillna(df_model['lag_2'])

X = df_model[FEATURES]
y = df_model['log_GDP']

# --- 5) Train-test split (time-aware simple holdout: last 3 years as test) ---
train_mask = df_model['Year_float'] <= (TRAIN_END_YEAR - 3)
X_train = X[train_mask]; y_train = y[train_mask]
X_val = X[~train_mask]; y_val = y[~train_mask]

# --- 6) Train LightGBM regressor ---
train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'seed': 42,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50)
    ]
)

# save model + encoder
joblib.dump(model, 'gdp_lgb_model.joblib')
joblib.dump(le, 'country_le.joblib')

# validate quick metric
pred_val = model.predict(X_val, num_iteration=model.best_iteration)
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_val, pred_val))
mae = mean_absolute_error(y_val, pred_val)
print(f"Validation RMSE (log space): {rmse:.4f}, MAE (log): {mae:.4f}")

# --- 7) Forecast population per country to 2030 (log-linear per country) ---
# We'll use countries from the merged df (i.e., those present)
countries = df['Country Name'].unique().tolist()
pop_forecasts = []

for country in countries:
    p = pop[pop['Country Name'] == country].sort_values('Year')
    years = p['Year'].values
    pops = p['Population'].values
    # require minimum data
    if len(pops) >= MIN_YEARS_FOR_POP_FORECAST:
        # model on log(pop)
        lr = LinearRegression()
        Xp = years.reshape(-1,1)
        yp = np.log1p(pops)
        lr.fit(Xp, yp)
        future_years = np.arange(TRAIN_END_YEAR+1, PREDICT_TO+1)
        pred_logp = lr.predict(future_years.reshape(-1,1))
        pred_pop = np.expm1(pred_logp)
        # assemble
        for yr, val in zip(future_years, pred_pop):
            pop_forecasts.append({'Country Name': country, 'Country Code': p['Country Code'].iloc[0], 'Year': int(yr), 'Population': val})
    else:
        # fallback flat: use last known population
        last_pop = pops[-1]
        for yr in range(TRAIN_END_YEAR+1, PREDICT_TO+1):
            pop_forecasts.append({'Country Name': country, 'Country Code': p['Country Code'].iloc[0], 'Year': int(yr), 'Population': last_pop})

pop_forecast_df = pd.DataFrame(pop_forecasts)

# combine observed population (<= TRAIN_END_YEAR) + forecasts (> TRAIN_END_YEAR)
pop_combined = pd.concat([
    pop[['Country Name','Country Code','Year','Population']],
    pop_forecast_df
], ignore_index=True).sort_values(['Country Name','Year']).reset_index(drop=True)

# --- 8) Prepare iterative dataset for forecasting GDP 2023->2030 ---
# We'll need the most recent observed log_GDP for each country as lag_1/lag_2/lag_3
last_gdp_by_country = df.groupby('Country Name').apply(lambda g: g.sort_values('Year').iloc[-3:]).reset_index(drop=True)

# build a dict of last known log_GDP sequence for each country (ordered oldest->newest)
initial_lags = {}
for country in countries:
    subset = df[df['Country Name']==country].sort_values('Year')
    logs = subset['log_GDP'].tolist()
    # ensure length at least 3
    if len(logs) >= 3:
        initial_lags[country] = logs[-3:]  # [t-2, t-1, t]
    else:
        # pad with the earliest available repeated
        padded = [logs[0]]*(3-len(logs)) + logs
        initial_lags[country] = padded

# We'll construct iterative predictions per country, year by year
pred_rows = []
for country in tqdm(countries, desc="Forecast countries"):
    code = df[df['Country Name']==country]['Country Code'].iloc[0]
    country_id = int(le.transform([code])[0])
    # get population series for this country from pop_combined
    pseries = pop_combined[pop_combined['Country Name']==country].set_index('Year')['Population'].to_dict()
    # get initial lag values (log_GDP)
    lags = initial_lags[country].copy()  # list of length 3: [t-2, t-1, t]
    for year in range(TRAIN_END_YEAR+1, PREDICT_TO+1):
        # prepare features:
        log_pop = np.log1p(pseries.get(year, pseries.get(TRAIN_END_YEAR)))  # fallback to last known
        features = {
            'Year_float': year,
            'log_Pop': log_pop,
            'lag_1': lags[-1],
            'lag_2': lags[-2],
            'lag_3': lags[-3],
            'rolling_3': np.mean(lags[-3:]),
            'country_id': country_id
        }
        X_pred = pd.DataFrame([features])[FEATURES]
        pred_log_gdp = model.predict(X_pred, num_iteration=model.best_iteration)[0]
        # store
        pred_gdp = np.expm1(pred_log_gdp)  # back-transform
        pred_rows.append({
            'Country Name': country,
            'Country Code': code,
            'Year': year,
            'Predicted_GDP': pred_gdp,
            'Predicted_log_GDP': pred_log_gdp
        })
        # update lags
        lags.append(pred_log_gdp)
        lags.pop(0)

pred_df = pd.DataFrame(pred_rows)

# --- 9) Extract 2030 predictions and rank top 10 ---
pred_2030 = pred_df[pred_df['Year']==2030].sort_values('Predicted_GDP', ascending=False).reset_index(drop=True)
top10_2030 = pred_2030.head(10)
print("Top 10 predicted GDP countries in 2030:")
print(top10_2030[['Country Name','Country Code','Predicted_GDP']].to_string(index=False))

# --- Create combined dataset: historical (1980-2022) + predictions (2023-2030) ---
# Prepare historical data in same format
historical_df = df[['Country Name', 'Country Code', 'Year', 'GDP']].copy()
historical_df.rename(columns={'GDP': 'Predicted_GDP'}, inplace=True)
historical_df['Predicted_log_GDP'] = df['log_GDP']

# Combine historical and predictions
combined_df = pd.concat([
    historical_df,
    pred_df
], ignore_index=True).sort_values(['Country Name', 'Year']).reset_index(drop=True)

# Save results
pred_df.to_csv("gdp_predictions_2023_2030.csv", index=False)
top10_2030.to_csv("top10_gdp_2030.csv", index=False)
combined_df.to_csv("gdp_combined_1980_2030.csv", index=False)
print(f"\nSaved combined historical + predictions to 'gdp_combined_1980_2030.csv'")

# --- 10) Quick diagnostic plot for top 5 countries (observed + forecast) ---
import matplotlib.pyplot as plt
top5 = top10_2030['Country Name'].tolist()[:5]

plt.figure(figsize=(14,8))
for country in top5:
    # Get observed data
    obs = df[df['Country Name']==country][['Year','GDP']].dropna().sort_values('Year')
    # Get forecast data
    f = pred_df[pred_df['Country Name']==country][['Year','Predicted_GDP']].sort_values('Year')
    
    # Plot observed data (solid line)
    plt.plot(obs['Year'], obs['GDP'], 'o-', linewidth=2, markersize=3, label=f"{country} observed")
    
    # Plot forecast data (dashed line) - include last observed point for smooth connection
    if len(obs) > 0:
        last_year = obs['Year'].iloc[-1]
        last_gdp = obs['GDP'].iloc[-1]
        # Create connected forecast line by including the last observed point
        forecast_years = [last_year] + f['Year'].tolist()
        forecast_gdp = [last_gdp] + f['Predicted_GDP'].tolist()
        plt.plot(forecast_years, forecast_gdp, 's--', linewidth=2, markersize=3, label=f"{country} forecast")
    else:
        plt.plot(f['Year'], f['Predicted_GDP'], 's--', linewidth=2, markersize=3, label=f"{country} forecast")

plt.yscale('log')
plt.legend(loc='best', fontsize=9)
plt.title("Observed GDP (1980-2022) and Forecast (2023-2030) for Top 5 Countries", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("GDP (US$, log scale)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
