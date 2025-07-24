import os
import numpy as np
import cv2
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import seaborn as sns
import joblib
import pandas as pd

# Paths
path = r"Dataset"
model_folder = "model"
os.makedirs(model_folder, exist_ok=True)

# Categories
categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# Load or generate feature arrays
X_file = os.path.join(model_folder, "X.txt.npy")
Y_file = os.path.join(model_folder, "Y.txt.npy")

if os.path.exists(X_file) and os.path.exists(Y_file):
    X = np.load(X_file)
    Y = np.load(Y_file)
    print("X and Y arrays loaded successfully.")
else:
    X, Y = [], []
    for category in categories:
        category_path = os.path.join(path, category)
        for file in os.listdir(category_path):
            if file.endswith(('.jpg', '.png')) and 'Thumbs.db' not in file:
                img_path = os.path.join(category_path, file)
                img_array = cv2.imread(img_path)
                if img_array is not None:
                    img_resized = resize(img_array, (64, 64, 3))
                    X.append(img_resized.flatten())
                    Y.append(categories.index(category))
    X = np.array(X)
    Y = np.array(Y)
    np.save(X_file, X)
    np.save(Y_file, Y)

# Count plot
sns.countplot(x=Y)
plt.title("Category Distribution")
plt.xlabel("Class Index")
plt.ylabel("Count")
plt.show()

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=77)
labels = categories

# Metrics storage
precision = []
recall = []
fscore = []
accuracy = []

def calculateMetrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    print(f"\n{algorithm} Metrics:")
    print(f"Accuracy: {a:.2f}%")
    print(f"Precision: {p:.2f}%")
    print(f"Recall: {r:.2f}%")
    print(f"F1 Score: {f:.2f}%")

    report = classification_report(testY, predict, target_names=labels)
    print(f"\n{algorithm} Classification Report:\n{report}")

    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="Blues", fmt="g")
    plt.title(f"{algorithm} Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

# Decision Tree Classifier
dt_model_file = os.path.join(model_folder, "DT_Model.pkl")
if os.path.exists(dt_model_file):
    dt_classifier = joblib.load(dt_model_file)
else:
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(x_train, y_train)
    joblib.dump(dt_classifier, dt_model_file)
    print("Decision Tree Classifier trained and saved.")

dt_predict = dt_classifier.predict(x_test)
calculateMetrics("DecisionTreeClassifier", dt_predict, y_test)

# Random Forest Classifier
rf_model_file = os.path.join(model_folder, "RFC_Model.pkl")
if os.path.exists(rf_model_file):
    rf_classifier = joblib.load(rf_model_file)
else:
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(x_train, y_train)
    joblib.dump(rf_classifier, rf_model_file)
    print("Random Forest Classifier trained and saved.")

rf_predict = rf_classifier.predict(x_test)
calculateMetrics("RandomForestClassifier", rf_predict, y_test)

# Prediction on new images
def predict_and_show(image_path, classifier, title):
    img = imread(image_path)
    img_resized = resize(img, (64, 64, 3))
    img_flattened = [img_resized.flatten()]
    output_number = classifier.predict(img_flattened)[0]
    output_name = categories[output_number]

    plt.imshow(img)
    plt.title(f"{title}: {output_name}")
    plt.axis('off')
    plt.show()

predict_and_show(r"dataset/glaucoma/_0_4517448.jpg", rf_classifier, "Predicted Output of RFC")
predict_and_show(r"dataset/normal/84_left.jpg", rf_classifier, "Predicted Output of RFC")

# Algorithm performance summary
columns = ["Algorithm Name", "Precision", "Recall", "F1 Score", "Accuracy"]
algorithm_names = ['DecisionTreeClassifier', 'RandomForestClassifier']
values = []

for i in range(len(algorithm_names)):
    values.append([algorithm_names[i], precision[i], recall[i], fscore[i], accuracy[i]])

summary_df = pd.DataFrame(values, columns=columns)
print("\nPerformance Summary:")
print(summary_df)
