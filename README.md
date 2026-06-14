# AUTOMOBILE_PRICE_PREDICTION
Predicting car prices using regression models based on vehicle specifications like engine size, horsepower, and curb weight
📌 Problem Statement

Given a set of automobile specifications, predict the market price of the car. This is a supervised regression problem where the model learns the relationship between vehicle features and price.


📂 Dataset

PropertyDetailSourceUCI Machine Learning Repository — Automobile DatasetRecords~205 rowsFeatures26 (15 numeric, 11 categorical)Targetprice (continuous)Missing valuesYes — stored as ? placeholders


🔧 Project Workflow

Data Loading → Cleaning → Encoding → Feature Selection → EDA → Modelling → Evaluation

1. Data Cleaning


1. Replaced ? placeholders with NaN
2. Imputed numeric columns (bore, stroke, horsepower, peak_rpm, normalized_losses) with column mean
3. Imputed num_of_doors with mode (four)
4. Dropped rows where target (price) was missing


2. Feature Engineering


Applied LabelEncoder to all 11 categorical columns


3. Feature Selection


1. Used SelectKBest (Chi² scoring) to rank all 25 features
2. Selected top 10 features
3. After correlation analysis (heatmap), removed 3 low-correlation features: stroke, height, normalized_losses
4. Final feature set (7): curb_weight, length, horsepower, wheel_base, engine_size, width, bore


4. EDA


1. Skewness & kurtosis check — all features within acceptable range
2. Boxplots, distribution plots, residual plots
3. Correlation heatmap revealed strong multicollinearity between size-related features


5. Modelling


StandardScaler applied before all models
67/20 train-test split (random_state=42)



🤖 Models & Results

ModelNotesLinear RegressionBaselineRidge RegressionL2 regularisation, tuned with GridSearchCVSVRSupport Vector Regression, tuned with GridSearchCVPCA + Linear RegressionCompressed to 2 principal componentsPCA-Linear ComboBest performer — top 5 PCA-contributing features


Best Model: PCA-Linear Combo using features curb_weight, length, wheel_base, horsepower, width



The PCA-guided approach outperformed all others by removing redundant correlated features (size-related), reducing noise and improving generalisation.


📊 Key Findings


1. curb_weight is the single strongest predictor of price
2. Strong multicollinearity exists between curb_weight, length, width, engine_size, and wheel_base — they all measure overall vehicle size
3. Linear models outperformed SVR, confirming the price-feature relationships are largely linear
4. Reducing features from 7 → 5 improved performance, showing that fewer but cleaner features beat more noisy ones



🛠️ Tech Stack

pandas · numpy · matplotlib · seaborn
sklearn (LinearRegression, Ridge, SVR, PCA, SelectKBest, StandardScaler, GridSearchCV)


📁 File Structure

automobile-price-prediction/
│
├── auto_imports.csv                        # Raw dataset
├── auto_imports_cleaned.csv                # Cleaned & encoded dataset
├── automobile_price_predict.ipynb          # Main notebook
├── best_model.pkl                          # Saved best model
│
└── plots/
    ├── boxplot_features.png
    ├── distribution_features.png
    ├── heatmap_features.png
    ├── pairplot_features.png
    ├── residual_plots.png
    ├── pca_components.png
    ├── lr_actual_vs_pred.png
    └── model_comparison.png


🚀 How to Run


Clone the repository


bashgit clone https://github.com/yourusername/automobile-price-prediction.git
cd automobile-price-prediction


Install dependencies


bashpip install pandas numpy matplotlib seaborn scikit-learn


Open the notebook


bashjupyter notebook automobile_price_predict.ipynb


Or upload directly to Google Colab and run all cells.




📈 Future Improvements


1. Apply log transformation to price (right-skewed target)
2. Try XGBoost / LightGBM for potential accuracy gains
3. Use k-fold cross-validation instead of a single train-test split
4. Engineer interaction features (e.g. horsepower-to-weight ratio)
