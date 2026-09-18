# 🏠 House Price Prediction

A machine learning project that predicts house prices based on housing-related features using **Linear Regression**.

## 🎯 Objective

The objective of this project is to predict the median value of houses using features such as median income, house age, number of rooms, population, occupancy, and geographical location.

## 📊 Dataset

This project uses the **California Housing Dataset** available through Scikit-learn.

The dataset contains **20,640 housing records** with 8 input features and 1 target variable.

### Features

* Median Income
* House Age
* Average Rooms
* Average Bedrooms
* Population
* Average Occupancy
* Latitude
* Longitude

### Target

**MedHouseVal** – Median house value.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## 🔄 Machine Learning Workflow

```text
Housing Dataset
      ↓
Data Exploration
      ↓
Missing Value Check
      ↓
Feature & Target Separation
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Linear Regression
      ↓
Price Prediction
      ↓
Model Evaluation
```

## 🧠 Machine Learning Model

### Linear Regression

Linear Regression is used to learn the relationship between housing features and house prices.

The model is trained using the training data and then used to predict prices for unseen test data.

## 📈 Model Evaluation

The model is evaluated using:

* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

### Results

```text
Mean Squared Error (MSE): YOUR_VALUE
Root Mean Squared Error (RMSE): YOUR_VALUE
R² Score: YOUR_VALUE
```

Replace `YOUR_VALUE` with the values from your terminal output.

## 📊 Visualization

The project generates an **Actual vs Predicted House Prices** scatter plot to compare the model's predictions with the actual values.

## 🏡 Example Prediction

The program allows users to enter housing details such as:

```text
Median income
House age
Average rooms
Average bedrooms
Population
Average occupancy
Latitude
Longitude
```

The trained model then predicts the estimated house value.

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── house_price_prediction.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/VK241105/House-Price-Prediction.git
```

### 2. Open the project folder

```bash
cd House-Price-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python house_price_prediction.py
```

## 💡 Skills Demonstrated

* Handling tabular data
* Data exploration
* Data preprocessing
* Feature scaling
* Regression
* Linear Regression
* Model evaluation
* Data visualization
* Python programming

## 👩‍💻 Author

**Vaishnavi Mane**

B.Tech CSE (Artificial Intelligence & Machine Learning)
Kolhapur Institute of Technology
