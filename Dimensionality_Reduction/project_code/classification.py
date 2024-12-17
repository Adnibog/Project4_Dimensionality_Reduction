import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from load_data import load_images_from_folder
from pca import PCA
from linear_regression_custom import LinearRegressionCustom
from additional_methods import additional_methods

def classify_faces(data_folder):
    images, labels = load_images_from_folder(data_folder)
    print(f"Loaded {len(images)} images.")

    # Reshape images to 2D array
    n_samples, h, w = images.shape
    X = images.reshape(n_samples, h * w)

    # Split into train and test sets
    split_ratio = 0.8
    split_index = int(n_samples * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]

    # Apply PCA
    pca = PCA(n_components=50)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    classifiers = {
        "Custom Linear Regression": LinearRegressionCustom(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f'{name} Accuracy: {accuracy * 100:.2f}%')

    # Apply additional methods
    X_train_lda, X_test_lda, X_train_ica, X_test_ica, X_train_tsne, X_test_tsne = additional_methods(data_folder)

    # Save the results
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/classification_results.txt', 'w') as f:
        for name, accuracy in results.items():
            f.write(f'{name} Accuracy: {accuracy * 100:.2f}%\n')
        f.write(classification_report(y_test, y_pred, target_names=[str(i) for i in np.unique(labels)]))

if __name__ == "__main__":
    data_folder = '/home/gobinda/My_Courses/QF/Project4/Dimensionality_Reduction/Data/att_faces'
    classify_faces(data_folder)
    