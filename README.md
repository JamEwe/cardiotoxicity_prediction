# Cardiotoxicity prediction

## Motivation
Drug candidates often cause a blockage of the potassium ion channel of the human ether-a-go-go-related gene (hERG). The blockage leads to long QT syndrome, which is a severe life-threatening cardiac side effect. Predict drug-induced hERG-related cardiotoxicity could facilitate drug discovery by filtering out toxic drug candidates.

## About 

Our machine learning model predicts cardiotoxicity of compound and shows what is contributing to it (explanation). It returns pIC50 value in moles. We checked and compared various approaches (based on molecular graph and MACCs fingerprint) and models. Model interpretation has been checked by LIME and Grad-CAM.

## Models

All model avaliable at: https://drive.google.com/file/d/1K7_SDzy-mRb6c9E-xTgHL7x_bMf06ftT/view?usp=sharing

To run app place rf.pkl and graph_model_128.pt files in the App folder.


## Results

| Model | MAE | MSE | RMSE | R2 Square |
| ------------- | ------------- | ------------- | ------------- |------------- |
| Ridge Regression | 0.611536 | 0.682949 | 0.826408 | 0.237518 |
| Random Forest Regressor | 0.430262 | 0.367253 | 0.606014 | 0.589979 |
| XGBoost  | 0.440287 | 0.382236  | 0.618252  | 0.573251  |
| LightGBM |	0.453793 |	0.402501 | 0.634430 |	0.550626 |
| HistGradient Boosting Regressor |	0.464982 |	0.408647 |	0.639255 |	0.543764 |
| Support Vector Regression |	0.425477 |	0.388762 |	0.623508 |	0.565965 |
| Averaging base models |	0.428720 |	0.366198 |	0.605143 |	0.591156 |
| Base models + Meta-model |	0.419492 |	0.363959 |	0.603290 | 0.593656 |
| Base models + Meta-model + cv |	0.415285 |	0.352917 |	0.594068 |	0.605984 |
| Fully Connected Neural Network |	0.421404 |	0.35799|	0.598331 |	0.600309 |
| Graph Convolutional Neural Network |	0.393488 |	0.323231 |	0.568534 |	0.639127 |
| GCNN + RFR |	0.377588 |	0.292669 |	0.540989 |	0.673248 |

## App
[![app_video](https://raw.githubusercontent.com/JamEwe/cardiotoxicity_prediction/master/images/app.png)](https://www.youtube.com/watch?v=AfpmEKPNqfM "Cardiotoxicity prediction app")


