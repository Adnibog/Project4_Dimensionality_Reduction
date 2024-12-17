import os
import time
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_images_from_folder
from pca import PCA
from linear_regression_custom import LinearRegressionCustom
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from additional_methods import additional_methods

def print_metrics(y_true, y_pred, method_name):
    accuracy = np.mean(y_true == y_pred) * 100
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"{method_name} Accuracy: {accuracy:.2f}%")
    print(f"{method_name} MSE: {mse:.2f}, MAE: {mae:.2f}")

def reconstruct_images(pca, images, image_shape, n_components_list, figures_dir):
    for n_components in n_components_list:
        pca.n_components = n_components
        transformed_images = pca.transform(images)
        reconstructed_images = pca.inverse_transform(transformed_images)
        selected_images = [reconstructed_images[i] for i in range(0, len(images), len(images)//10)]
        plt.figure(figsize=(15, 6))
        for i, image in enumerate(selected_images):
            plt.subplot(2, 5, i + 1)
            plt.imshow(image.reshape(image_shape), cmap='gray')
            plt.title(f"n={n_components}, Image {i+1}")
            plt.axis('off')
        plt.savefig(os.path.join(figures_dir, f'reconstructed_images_n_{n_components}.png'))
        plt.close()

def plot_cumulative_variance(pca, figures_dir):
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs Number of Principal Components')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'cumulative_variance.png'))
    plt.close()

def train_and_evaluate(clf, name, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy * 100:.2f}%')
    return name, accuracy

def main():
    start_time = time.time()
    data_folder = '/home/gobinda/My_Courses/QF/Project4/Dimensionality_Reduction/Data/att_faces'
    images, labels = load_images_from_folder(data_folder)
    print(f"Loaded {len(images)} images in {time.time() - start_time:.2f} seconds.")

    # Reshape images to 2D array
    n_samples, h, w = images.shape
    X = images.reshape(n_samples, h * w)

    # Split into train and test sets
    split_ratio = 0.8
    split_index = int(n_samples * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]
    print(f"Data split into train and test sets in {time.time() - start_time:.2f} seconds.")

    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(data_folder, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Apply PCA
    pca_start_time = time.time()
    pca = PCA(n_components=100)  # Adjust the number of components if needed
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA completed in {time.time() - pca_start_time:.2f} seconds.")

    # Plot leading eigenfaces
    pca.plot_eigenfaces((h, w))

    # Plot cumulative variance
    plot_cumulative_variance(pca, figures_dir)

    # Reconstruct images for different numbers of principal components
    n_components_list = [50, 75, 100]
    reconstruct_images(pca, X_test, (h, w), n_components_list, figures_dir)

    classifiers = {
        "Custom Linear Regression": LinearRegressionCustom(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    training_start_time = time.time()
    results = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(clf, name, X_train_pca, y_train, X_test_pca, y_test) for name, clf in classifiers.items())
    results = dict(results)
    print(f"Classifier training and evaluation completed in {time.time() - training_start_time:.2f} seconds.")

    # Apply additional methods
    additional_methods_start_time = time.time()
    additional_methods(data_folder)
    print(f"Additional methods completed in {time.time() - additional_methods_start_time:.2f} seconds.")

    # Save the results
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/classification_results.txt', 'w') as f:
        for name, accuracy in results.items():
            f.write(f'{name} Accuracy: {accuracy * 100:.2f}%\n')
    print(f"Results saved in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
    