🌱 Fertilizer Recommendation System
A machine learning–powered web application that recommends the most suitable fertilizer based on soil properties, crop type, environmental conditions, and farming practices. Built with Scikit-learn, XGBoost, and deployed via a Streamlit interface.

📌 Project Overview
Agriculture productivity depends heavily on using the right fertilizer at the right time. This project automates that decision using supervised machine learning. Given a set of soil and crop inputs, the system predicts one of seven fertilizer categories with high accuracy.
Predicted Fertilizer Classes:
LabelFertilizer0MOP1Urea2Zinc Sulphate3Compost4NPK5DAP6SSP

🗂️ Project Structure
├── mlproject.ipynb               # Full ML pipeline: EDA, preprocessing, training, evaluation
├── app.py                        # Streamlit web application
├── svm_model_original.pkl        # Trained SVM model (saved via joblib)
├── scaler.pkl                    # Fitted StandardScaler (saved via joblib)
├── fertilizer_recommendation_Shiva.csv   # Dataset
└── README.md

📊 Dataset
File: fertilizer_recommendation_Shiva.csv
The dataset contains labeled records linking soil/crop/environmental conditions to an appropriate fertilizer. After initial preprocessing, the following columns were dropped due to irrelevance or data quality concerns:

Organic_Carbon
Electrical_Conductivity
Crop_Growth_Stage
Previous_Crop
Region
Yield_Last_Season
Fertilizer_Used_Last_Season

Final Feature Set (22 features after encoding):
TypeFeaturesNumericalSoil_pH, Soil_Moisture, Nitrogen_Level, Phosphorus_Level, Potassium_Level, Temperature, Humidity, RainfallCategoricalSoil_Type (Clay, Silt, Sandy, Loamy), Crop_Type (Cotton, Maize, Wheat, Potato, Rice, Sugarcane, Tomato), Season (Kharif, Zaid, Rabi), Irrigation_Type (Canal, Sprinkler, Rainfed, Drip)

🔬 ML Pipeline (mlproject.ipynb)
1. Data Preprocessing

Dropped irrelevant columns
Stripped whitespace from column names
Checked for null values, infinite values, and duplicates
Inspected unique values in categorical columns

2. Exploratory Data Analysis (EDA)

Count plots for fertilizer labels, season distribution, crop types, and rainfall
Correlation heatmap to identify multicollinearity
Boxplots for outlier detection across all numerical features
Pairplot colored by fertilizer class for visual class separability
Dropped highly correlated features (threshold > 0.9)

3. Encoding & Scaling

One-hot encoding applied to all categorical columns using pd.get_dummies() with drop_first=True
Features scaled using StandardScaler
Target labels encoded using LabelEncoder

4. Dimensionality Reduction (Exploratory)

PCA was applied (3 components) for visualization purposes
A scree plot and 2D scatter were generated to assess variance explained
PCA-reduced data was also used for model comparison (but not selected for final deployment)

5. Train-Test Split

80/20 train-test split with random_state=42
Separate splits maintained for original and PCA-reduced data

6. Model Training with GridSearchCV
Seven classifiers were evaluated using 5-fold cross-validated GridSearchCV:
ModelHyperparameters TunedNaive Bayesvar_smoothingRandom Forestn_estimators, max_depth, min_samples_split, min_samples_leafSVMC, kernel, gammaKNNn_neighbors, weights, metricLogistic RegressionC, solverXGBoostn_estimators, max_depth, learning_rate, subsampleExtra Treesn_estimators, max_depth, min_samples_split
7. Evaluation Metrics
Each model was evaluated on Accuracy, Precision (weighted), Recall (weighted), and F1-score (weighted), along with a confusion matrix.

🏆 Results
Top performing models on the original (non-PCA) data:
ModelAccuracyF1-ScoreSVM1.001.000Logistic Regression1.001.000XGBoost1.001.000

⚠️ Note: Perfect accuracy on the test set may indicate overfitting or a relatively clean/structured dataset. It is recommended to validate with a held-out external validation set before production deployment.

Chosen Model for Deployment: SVM (RBF kernel) trained on original scaled data.

💾 Saved Artifacts
After training, two files are saved using joblib:

scaler.pkl — The StandardScaler fitted on training data. Used to transform inputs before prediction.
svm_model_original.pkl — The best-performing SVM classifier.


🖥️ Web Application (app.py)
The Streamlit app provides a user-friendly interface to get fertilizer predictions in real time.
Features

Input fields for all 12 raw features (numerical + categorical)
Automatic one-hot encoding and feature alignment to match training columns
Scaling using the saved StandardScaler
Prediction using the saved SVM model
Human-readable fertilizer name displayed as output

Input Fields
FieldTypeRange / OptionsSoil pHNumeric0.0 – 14.0Soil Moisture (%)Numeric0.0 – 100.0NitrogenNumeric0.0 – 300.0PhosphorusNumeric0.0 – 300.0PotassiumNumeric0.0 – 300.0Temperature (°C)Numeric0.0 – 60.0Humidity (%)Numeric0.0 – 100.0Rainfall (mm)Numeric0.0 – 5000.0Soil TypeCategoricalClay, Silt, Sandy, LoamyCrop TypeCategoricalCotton, Maize, Wheat, Potato, Rice, Sugarcane, TomatoSeasonCategoricalKharif, Zaid, RabiIrrigation TypeCategoricalCanal, Sprinkler, Rainfed, Drip

⚙️ Installation & Setup
Prerequisites

Python 3.8+
pip

1. Clone the Repository
git clone https://github.com/Yummy006/fertilizer-recommendation-system.git
cd fertilizer-recommendation-system
2. Install Dependencies
bashpip install -r requirements.txt
Key dependencies:
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
seaborn
matplotlib
3. Train the Model (Optional)
If you want to retrain the model from scratch, run the Jupyter notebook:
bashjupyter notebook mlproject.ipynb
This will generate svm_model_original.pkl and scaler.pkl.
4. Run the App
bashstreamlit run app.py
The app will open in your browser at http://localhost:8501.

📁 Requirements
Make sure the following files are present in the project root before running the app:

svm_model_original.pkl
scaler.pkl

These are generated by running the notebook. If they are not present, the app will throw a FileNotFoundError on startup.

🔮 Future Improvements

Add cross-validation results and learning curves to better assess generalization
Integrate real soil sensor data for live predictions
Extend the crop list and fertilizer categories
Add SHAP-based feature importance explanations in the app
Deploy on cloud platforms (Streamlit Cloud, Hugging Face Spaces, or AWS)


