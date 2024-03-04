import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

# Function to load data with different encodings
def load_data(uploaded_file):
    if uploaded_file is None or uploaded_file.size == 0:
        st.error("No file uploaded or the uploaded file is empty.")
        return None

    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            if not df.empty:
                return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.warning(f"An error occurred with {encoding} encoding: {e}")

    try:
        df = pd.read_excel(uploaded_file)
        if not df.empty:
            return df
    except Exception as e:
        st.error(f"Failed to read as Excel: {e}")
        return None

    st.error("Unable to parse the file with tried encodings and as Excel. Please check the file format.")
    return None

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def compute_metrics(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    # For multiclass, we use 'macro' or 'weighted' average
    sensitivity = recall_score(y_test, predictions, average='macro')
    specificity = recall_score(y_test, predictions, average='macro') # Note: Specificity is not directly supported

    return accuracy, cm, sensitivity,specificity
def compute_class_specific_metrics(y_true, y_pred, class_labels):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    metrics = {}

    for i, label in enumerate(class_labels):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum().sum() - TP - FP - FN

        # Sensitivity (Recall) and Specificity for each class
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        metrics[label] = {'Sensitivity': sensitivity, 'Specificity': specificity}
    
    return metrics
def compute_metrics_for_all_models(models, X_train, X_test, y_train, y_test):
    model_results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average='macro')

        # You can add more metrics as needed
        model_results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Recall": recall
            # Add other metrics here
        })

    return pd.DataFrame(model_results)
# Streamlit App
st.title('Machine Learning Model Comparison')

uploaded_file = st.file_uploader("Choose a CSV or Excel file")
if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data is not None:
        st.write("Data Loaded Successfully:")
        st.write(data)

        # Preprocess the data
        # Apply Label Encoding
        label_encoders = {col: LabelEncoder() for col in data.columns}
        for col in data.columns:
            data[col] = label_encoders[col].fit_transform(data[col].astype(str))

        # Assuming the last column is the target variable
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Convert all feature names in X to strings
        X.columns = X.columns.astype(str)

        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if st.button('Train and Evaluate Models'):
            models = {
                "Random Forest": RandomForestClassifier(),
                "SVM Linear": SVC(kernel='linear'),
                "SVM RBF": SVC(kernel='rbf'),
                "SVM Poly": SVC(kernel='poly'),
                "Naive Bayes": GaussianNB()
            }

            model_results_df = compute_metrics_for_all_models(models, X_train, X_test, y_train, y_test)
            
            st.subheader("Model Comparison Results")
            st.table(model_results_df)
            class_names = y.unique()

            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                class_metrics = compute_class_specific_metrics(y_test, predictions, class_names)

                st.subheader(f"{name} Results")
                for label, metrics in class_metrics.items():
                    st.text(f"Class {label} - Sensitivity: {metrics['Sensitivity']}, Specificity: {metrics['Specificity']}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, predictions))
                cm_fig = plot_confusion_matrix(confusion_matrix(y_test, predictions), class_names)
                st.pyplot(cm_fig)
                
                st.subheader("Model Performance Visualization")
                fig, ax = plt.subplots()
                model_results_df.plot(kind='bar', x='Model', y=['Accuracy', 'Recall'], ax=ax)
                st.pyplot(fig)
