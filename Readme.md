# <span style = "color: Orange;"> Fraud Detection in Financial Transcations Using Machine Learning / Deep Learning Models </span>
<hr style="height:1px; background-color:lightcoral; border:none;">

This project focuses on detecting fraudulent financial transactions using machine learning techniques to address the growing challenges posed by cybercrime in the digital economy. Financial transaction fraud—such as credit card fraud, identity theft, and account takeovers—results in substantial financial losses and erodes trust in financial institutions. To combat these evolving threats, advanced data-driven approaches are required that can identify suspicious patterns effectively and efficiently.

Using historical transaction data, the project implements a complete machine learning pipeline involving data preprocessing, feature engineering, and model training. Special attention is given to the inherent class imbalance in fraud detection problems, where fraudulent transactions represent a small minority of the data. Various sampling strategies and evaluation metrics are applied to improve the model’s ability to accurately detect fraudulent activities while minimizing false positives.

The ultimate objective of this project is to build a robust and reliable fraud detection model capable of predicting fraudulent transactions in real time. By improving early detection accuracy, the system aims to reduce financial losses, mitigate risks for businesses and consumers, and enhance overall trust in financial transaction systems.

# <span style = "color: #add8e6;"> 📊 Dataset</span> 

<hr style="height:1px; background-color: lightcoral; border:none;">

The dataset used in this project was sourced from Hugging Face. It contains historical credit card transactions from 2019 to 2020, including consumer and merchant information, transaction details, and fraud labels.

Dataset source: Hugging Face: Credit Card Transactions

- Total records: 1296675
- Fraudulent records: 7506
- Legitimate records: 1289169

Each transaction includes the following features:

trans_date_trans_time : Date and time of the transaction

cc_num : Consumer account number

merchant : Merchant name

category : Consumer category

amt : Transaction amount

first : Consumer first name

last : Consumer last name

gender : Consumer gender

street : Consumer street address

city : Consumer city

state : Consumer state

zip : Consumer postal code

lat : Consumer latitude

long : Consumer longitude

city_pop : City population

job : Consumer occupation

dob : Date of birth of the consumer

trans_num : Transaction number

unix_time : Transaction timestamp in Unix format

merch_lat : Merchant latitude

merch_long : Merchant longitude

is_fraud : Fraud label (0 = legitimate, 1 = fraudulent)

merch_zipcode : Merchant postal code





# <span style = "color: #add8e6;">  ✔️ Requirements </span> 

<hr style="height:1px; background-color: lightcoral; border:none;">

- Python 3.x
- Libraries:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - xgboost
    - tensorflow



# <span style = "color: #add8e6;">  ⚙️ Methodology </span> 

<hr style="height:1px; background-color: lightcoral; border:none;">


### 1. Exploratory Data Analysis

- Analyzed the dataset to compare fraudulent and non-fraudulent transactions.
- Analyzed the distribution of various features to identify patterns between fraud and non-fraud cases.

### 2. Data Preprocessing

- Evaluated the dataset for missing values to ensure data quality before modeling.
- Removed unnecessary columns and irrelevant data from the dataset.

### 3. Feature Engineering

- Computed the distribution of transactions over time to identify patterns between weekdays and weekends.

- Transformed categorical features into numerical representations using Label Encoding

- Split the dataset into training and testing sets.

### 4. Models

- XGBoost
- Artificial Neural Network(ANN) \- Feed Forward Neural Network

### 5.Evaluation Metrics

- Accuracy 
- F1 Score
- ROC-AUC


# <span style = "color: #add8e6;">   📈 Results </span> 

<hr style="height:1px; background-color: lightcoral; border:none;">


### XGBoost Model(Without Class Weights)

![XGBoost (without class weights)](images\xgboost_withoutclass.png)

### XGBoost Model(With Class Weights)

![XGBoost (with class weights)](images\xgboost_withclass.png)

### Artificial Neural Network (ANN) -  Feed Forward Neural Network

![ANN](images\ANN.png)






# <span style = "color: #add8e6;">   📍 Conclusion </span> 

<hr style="height:1px; background-color: lightcoral; border:none;">

By applying machine learning and neural network techniques, this project demonstrated effective detection of fraudulent transactions. Real-time transaction data from Hugging Face was preprocessed through exploratory data analysis, handling missing values, extracting date-time features, encoding categorical variables, and normalizing numerical features. The dataset was then split into training and testing sets (75% / 25%) and models including XGBoost (with and without class weights) and a feed-forward neural network (ANN) were trained.

Evaluation results:

- XGBoost (without class weight): Accuracy = 1.00, F1-score = 0.70, ROC-AUC = 0.97

- XGBoost (with class weight): Accuracy = 1.00, F1-score = 0.80, ROC-AUC = 0.99

- ANN model: Accuracy = 1.00, F1-score = 0.61, ROC-AUC = 0.94

The results highlight that addressing class imbalance and careful feature preprocessing can significantly enhance fraud detection performance. XGBoost with class weighting outperformed other models, underscoring the importance of model selection and evaluation metrics like F1-score and ROC-AUC in critical applications such as financial fraud prevention.