import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from load_data import load_images_from_folder
from pca import PCA 
import matplotlib.pyplot as plt
import os

def additional_methods(data_folder):
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

    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(data_folder, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Apply LDA
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    lda = LDA(n_components=min(n_features, n_classes - 1))
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    plt.figure()
    plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='viridis', marker='o')
    plt.title('LDA of Training Data')
    plt.savefig(os.path.join(figures_dir, 'lda_train.png'))
    plt.close()

    plt.figure()
    plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, cmap='viridis', marker='o')
    plt.title('LDA of Test Data')
    plt.savefig(os.path.join(figures_dir, 'lda_test.png'))
    plt.close()

    # Apply ICA
    ica = FastICA(n_components=min(n_features, n_classes - 1))
    X_train_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)

    plt.figure()
    plt.scatter(X_train_ica[:, 0], X_train_ica[:, 1], c=y_train, cmap='viridis', marker='o')
    plt.title('ICA of Training Data')
    plt.savefig(os.path.join(figures_dir, 'ica_train.png'))
    plt.close()

    plt.figure()
    plt.scatter(X_test_ica[:, 0], X_test_ica[:, 1], c=y_test, cmap='viridis', marker='o')
    plt.title('ICA of Test Data')
    plt.savefig(os.path.join(figures_dir, 'ica_test.png'))
    plt.close()

    # Apply custom PCA and plot cumulative variance
    pca = PCA(n_components=n_samples)
    pca.fit(X)
    X_pca = pca.transform(X)
    cumulative_variance = np.cumsum(pca.components[:n_samples].var(axis=0))

    plt.figure()
    plt.plot(cumulative_variance)
    plt.xlabel('Number of Eigenfaces')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance vs. Number of Eigenfaces')
    plt.savefig(os.path.join(figures_dir, 'cumulative_variance.png'))
    plt.close()

    # Display accuracy and results for other methods
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier()
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.2%}")
        print(f"{name} MSE: {mse:.2f}, MAE: {mae:.2f}")

        # Save the results plot
        plt.figure()
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o')
        plt.title(f'{name} Predictions')
        plt.savefig(os.path.join(figures_dir, f'{name.lower().replace(" ", "_")}_predictions.png'))
        plt.close()
