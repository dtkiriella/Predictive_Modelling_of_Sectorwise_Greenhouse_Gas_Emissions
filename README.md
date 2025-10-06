# Predictive_Modelling_of_Sectorwise_Greenhouse_Gas_Emissions

## Project Overview
Climate change remains one of the most critical challenges of the 21st century, driven by the continuous increase in greenhouse gas (GHG) emissions. This project aims to develop a **machine learning-based predictive model** to forecast **sector-wise GHG emissions** for the world’s top-emitting countries up to **2030**, assessing their alignment with the **2015 Paris Agreement** targets.

By integrating historical emissions data with socio-economic indicators such as **GDP** and **population**, this study provides a **data-driven framework** for evaluating global emission patterns, identifying high-risk sectors, and supporting future climate policy research.



## Project Objectives
1. **Forecast sector-wise GHG emissions** up to 2030 for the top-emitting countries using advanced ML algorithms.  
2. **Identify key sectors** contributing the most to emissions within each country.  
3. **Analyze the most dominant greenhouse gases** (CO₂, CH₄, N₂O, F-gases) across countries and sectors.  
4. **Evaluate global emission progress** relative to each country’s Nationally Determined Contributions (NDCs) under the 2015 Paris Agreement.  



## Methodology
This research follows a **quantitative, predictive modelling approach** that integrates data collection, feature engineering, model development, and evaluation.  
The workflow includes:

### 1. **Data Acquisition**
- **Emission Data (1990–2020):** Sector-wise and gas-wise GHG emissions from **Climate Watch**.  
- **GDP Data (1960–2024):** From **World Bank Open Data** (in USD).  
- **Population Data (1960–2024):** From **World Bank Open Data**.  

### 2. **Data Preprocessing**
- Cleaning and merging datasets.  
- Handling missing values and standardising units (MtCO₂e).  
- Generating lag features and growth rates for temporal trend capture.  

### 3. **Model Development**
Three models will be compared:
- **Extreme Gradient Boosting (XGBoost)** – efficient, interpretable tree-based model for complex nonlinear data.  
- **Random Forest (RF)** – ensemble approach to reduce overfitting and handle variable interactions.  
- **Advanced Long Short-Term Memory (LSTM)** – a deep learning model designed for multivariate time series forecasting.  

**Hyperparameter tuning:**  
- Performed using **Bayesian Optimisation** and **time-series cross-validation** to improve generalisability.  

**Evaluation Metrics:**  
- Mean Absolute Error (MAE)  
- Root Mean Square Error (RMSE)  
- Mean Absolute Percentage Error (MAPE)  
- Coefficient of Determination (R²)  

### 4. **Interpretability and Validation**
- Apply **SHAP (SHapley Additive exPlanations)** to interpret model outputs and identify key socioeconomic drivers.  
- Conduct **scenario and sensitivity analyses** (e.g., higher GDP growth, population surges).  
- Compare predicted emissions to each country’s **NDC targets**.
