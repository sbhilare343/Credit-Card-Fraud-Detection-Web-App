# Credit-Card-Fraud-Detection-Web-App

---
### **Dataset**
    https://drive.google.com/file/d/1savE2kJZIGa_z4-QHeiNDgrvsWH4kPMe/view?usp=drive_link
---

### 1. **Mounting Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

- This mounts your Google Drive to the Colab environment so you can easily access files from your Drive. It mounts to `/content/drive`.

---

### 2. **Importing Required Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

- **Pandas** and **NumPy** are used for data manipulation and numerical operations.
- **Matplotlib** and **Seaborn** are used for plotting and visualizing the data.

---

### 3. **Loading and Preprocessing Data**

```python
test_file_path = '/content/drive/My Drive/Customer Data/Copy of fraudTrain.csv'
data = pd.read_csv(test_file_path, nrows=10000)
```

- This loads the dataset `fraudTrain.csv` from Google Drive. You load only the first 10,000 rows for faster processing.

---

```python
data.fillna(method='ffill', inplace=True)
scaler = StandardScaler()
data[['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']] = scaler.fit_transform(data[['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']])
```

- Missing values are filled using forward fill (`ffill`).
- Some numeric columns (`amt`, `lat`, `long`, etc.) are normalized using `StandardScaler` for better model performance.

---

### 4. **Model Selection, Training, and Evaluation**

#### **One-Hot Encoding:**

```python
data = pd.get_dummies(data, columns=['merchant', 'category', 'gender'], drop_first=True)
```

- Converts categorical variables (`merchant`, `category`, `gender`) into numerical form by creating dummy/indicator variables using One-Hot Encoding.

#### **Feature Selection and Train-Test Split:**

```python
X = data.drop(columns=[...])  # Dropping irrelevant columns
y = data['is_fraud']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

- The dataset is split into features (`X`) and the target variable (`y`).
- Irrelevant columns (e.g., `first`, `last`, `dob`, `cc_num`, etc.) are dropped.
- The data is split into training (70%) and testing (30%) sets.

#### **Model Training and Evaluation:**

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

- A **Logistic Regression** model is trained with a maximum of 1000 iterations.
- Predictions are made on the test data.

```python
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

- The model's performance is evaluated using accuracy, a confusion matrix, and a classification report.

---

### 5. **Making Predictions on New Data**

```python
new_data = pd.DataFrame({
    ...
})
new_data[['amt', 'lat', ...]] = scaler.transform(new_data[['amt', 'lat', ...]])
```

- New data is passed into the model for prediction.
- The new data is normalized and one-hot encoded similarly to the training data.
  
```python
prediction = model.predict(new_data)
print(f'Prediction: {prediction}')
```

- The model predicts whether the new transaction is fraudulent (`1`) or not (`0`).

---

### 6. **Optional Features: Visualization**

#### **Confusion Matrix:**

```python
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
```

- A heatmap visualizes the confusion matrix to better understand the model's performance in classifying fraud vs. non-fraud.

#### **Feature Importance:**

```python
importance = model.coef_[0]
plt.barh(features, importance)
plt.title('Feature Importance')
```

- Visualizes feature importance in the Logistic Regression model, showing which features contribute most to the model's decision-making process.

---

### 7. **Experimenting with Different Models**

#### **Decision Tree and Random Forest:**

```python
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
print(accuracy_score(y_test, dt_model.predict(X_test)))

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
print(accuracy_score(y_test, rf_model.predict(X_test)))
```

- Two additional models, **Decision Tree** and **Random Forest**, are trained and their performance is compared using accuracy.

---

### 8. **Saving the Model and Scaler**

```python
import joblib
joblib.dump(rf_model, '/content/drive/My Drive/model.joblib')
joblib.dump(scaler, '/content/drive/My Drive/scaler.joblib')
```

- The **Random Forest** model and the scaler are saved to Google Drive using `joblib`.

---

### 9. **Deploying with Gradio**

```python
def predict_fraud(merchant, category, amt, gender, lat, long, city_pop, merch_lat, merch_long):
    ...
gr.Interface(fn=predict_fraud, inputs=inputs, outputs=outputs).launch()
```

- **Gradio** is used to create a simple web interface to interact with the fraud detection model.
- Inputs such as `merchant`, `category`, `amt`, etc., are taken from the user, and the prediction is displayed through the Gradio interface.
