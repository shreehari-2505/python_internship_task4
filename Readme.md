# 🎯 Logistic Regression – Binary Classification on Breast Cancer Dataset

## 📌 Objective

This project is part of my **AI & ML Internship – Task 4**.  
The goal is to **build a Binary Classifier using Logistic Regression** to predict whether a tumor is **Malignant (M)** or **Benign (B)** using the **Breast Cancer Wisconsin dataset**.

---

## 🛠 Tools & Libraries Used

| Tool / Library | Purpose |
| --- | --- |
| **Python** | Core programming language |
| **Pandas** | Data loading, cleaning, and wrangling |
| **NumPy** | Numerical computations, array operations |
| **Matplotlib/Seaborn** | Data visualization |
| **Scikit-learn** | Data preprocessing, model building, evaluation |
| **Joblib** | Model serialization |
| **Google Colab** | Notebook execution environment |

---

## 🔄 Workflow – Step-by-Step Logic Flow

Text-based flowchart showing entire process:

[Start]

↓

Load dataset (`pd.read_csv`)

↓

Initial inspection → shape, columns, first few rows

↓

Drop unnecessary/empty columns (`Unnamed: 32`, `id`)

↓

Check and clean target column (`diagnosis`) → ensure only 'M' or 'B'

↓

Encode target: 'M' → 1, 'B' → 0

↓

Split into features (X) and target (y)

↓

Train-Test Split (80/20) with class stratification

↓

Standardize features (mean=0, std=1) with `StandardScaler`

↓

Define and train Logistic Regression model (`LogisticRegression`)

↓

Predict on test set (`predict`) and get probabilities (`predict_proba`)

↓

Evaluate model performance → Confusion Matrix, Accuracy, Precision, Recall, ROC-AUC

↓

Plot ROC Curve and Feature Importance

↓

Adjust classification threshold for trade-off between precision \& recall

↓

Save trained model and scaler with Joblib for reuse

↓

[End]




---

## 🧪 Steps Performed in Detail

### 1️⃣ **Data Loading**
- Dataset: **Breast Cancer Wisconsin (Diagnostic) Data Set**
- Loaded using:   df = pd.read_csv("data.csv")


### 2️⃣ **Data Cleaning**
- Dropped **useless/null** columns:  df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

- Filtered `diagnosis` to **M**/**B** only, mapped to 1 and 0.

### 3️⃣ **EDA – Exploratory Data Analysis**
- Checked dataset shape & missing values.
- Countplot for target class distribution.
- Summary statistics for quick feature overview.

### 4️⃣ **Preprocessing**
- Features (`X`) and Target (`y`) separation.
- Split into Train/Test set:  train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

- Standardized features with: StandardScaler()

### 5️⃣ **Model Training**
- Defined Logistic Regression model:
- LogisticRegression(solver='lbfgs', max_iter=1000)

- Fitted with: model.fit(X_train_scaled, y_train)


### 6️⃣ **Predictions & Evaluation**
- Predictions (`predict`) and Probabilities (`predict_proba`).
- Computed:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- ROC-AUC Score

- Plotted ROC Curve for model discrimination power.

### 7️⃣ **Feature Importance**
- Extracted model coefficients and plotted top features influencing classification.

### 8️⃣ **Threshold Tuning**
- Changed decision threshold from default (0.5) to custom (e.g. 0.3) to check trade-offs.
- Observed impact on Recall & Precision.

### 9️⃣ **Model Saving**
- Saved model & scaler:

joblib.dump(model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')





---

## 📚 Vocabulary of Functions & Commands Used

| Command / Function | Purpose |
| --- | --- |
| `pd.read_csv(path)` | Reads a CSV file into pandas DataFrame |
| `df.head()` | Displays first few rows of DataFrame |
| `df.shape` | Shows dataset dimensions `(rows, columns)` |
| `df.drop(columns, axis, inplace)` | Removes unwanted columns |
| `df['col'].map(mapping_dict)` | Maps categorical values to numbers |
| `train_test_split(X, y, test_size, stratify, random_state)` | Splits dataset into train and test sets |
| `StandardScaler()` | Scales features to mean=0, std=1 |
| `.fit_transform(data)` | Fits scaler to data and transforms |
| `.transform(data)` | Transforms new data using fitted scaler |
| `LogisticRegression()` | Defines logistic regression model |
| `.fit(X_train, y_train)` | Trains model on training set |
| `.predict(X_test)` | Predicts labels for new data |
| `.predict_proba(X_test)` | Gets probabilities for predictions |
| `confusion_matrix(y_true, y_pred)` | Creates confusion matrix |
| `classification_report(y_true, y_pred)` | Generates precision, recall, f1-score |
| `roc_auc_score(y_true, y_proba)` | Computes ROC-AUC score |
| `roc_curve(y_true, y_proba)` | Generates points for ROC plot |
| `sns.heatmap()` | Plots heatmap (used for confusion matrix) |
| `plt.plot()` | Plots graphs in matplotlib |
| `joblib.dump(object, filename)` | Saves Python object to file |
| `joblib.load(filename)` | Loads saved object from file |

---

## 📊 Key Insights
- Logistic regression is effective for binary classification when features are well-scaled.
- ROC-AUC score of ~**0.996** indicates **excellent separability** between classes.
- Feature scaling dramatically improves convergence.
- Custom threshold tuning can optimize for recall/precision depending on project needs (critical in medical diagnosis).

---

## ✍ Prepared By
📄 README prepared by **Perplexity AI**  
