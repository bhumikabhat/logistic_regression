# Logistic Regression Binary Classification - Breast Cancer Dataset

## 📌 Objective
Build a **binary classifier** using Logistic Regression to predict whether a tumor is **malignant (M)** or **benign (B)** based on diagnostic features from the Breast Cancer Wisconsin dataset.

## 🛠 Tools & Libraries
- **Python 3**
- **Pandas** – Data loading & preprocessing  
- **Scikit-learn** – Model training & evaluation  
- **Matplotlib** – Visualization (ROC curve)  

## 📂 Dataset
The dataset used is `data.csv`, which contains:
- **Features:** 30 numeric columns describing tumor measurements.
- **Target (`diagnosis`):**  
  - `M` → Malignant (encoded as `1`)  
  - `B` → Benign (encoded as `0`)  

Extra columns `id` and `Unnamed: 32` were dropped.

## 🔍 Steps Followed

1. **Load Dataset**  
   Read `data.csv` and inspect columns, missing values, and data types.

2. **Data Preprocessing**  
   - Drop unused columns: `id`, `Unnamed: 32`
   - Encode target variable: `M` → 1, `B` → 0
   - Split into train/test sets (80/20 split)
   - Standardize features using `StandardScaler`

3. **Model Training**  
   Train a **Logistic Regression** model using scikit-learn.

4. **Evaluation Metrics**  
   - **Confusion Matrix**
   - **Precision, Recall, F1-score**
   - **ROC-AUC Score**
   - **ROC Curve Plot**

5. **Sigmoid Function Explanation**  
   Logistic Regression uses the **sigmoid** function to map any real number `z` into a probability between 0 and 1:  
   \[
   \sigma(z) = \frac{1}{1 + e^{-z}}
   \]
   This probability is then compared against a threshold (default 0.5) to assign the class label.

## 📊 Results
- **Confusion Matrix:**
[[70, 1],
[ 2, 41]]

markdown
Copy
Edit
- **Accuracy:** ~97%
- **ROC-AUC:** ~0.997 (excellent model performance)

## 📈 ROC Curve
The ROC curve shows the trade-off between **True Positive Rate** and **False Positive Rate**, with an AUC close to 1.0 indicating excellent separability.

## 🚀 How to Run
1. Install dependencies:
 ```bash
 pip install pandas scikit-learn matplotlib
Place data.csv in the same directory as the script.

Run the Python file:

bash
Copy
Edit
python logistic_regression.py
View the printed metrics and the plotted ROC curve.

📬 Author
Bhumika Bhat – AI/ML Enthusiast & Developer

yaml
Copy
Edit

---
