import matplotlib.pyplot as plt
from load_data import load_images_from_folder
from pca import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import numpy as np

def reconstruct_faces(data_folder):
    images, labels = load_images_from_folder(data_folder)
    if images.size == 0:
        print("No images loaded. Exiting.")
        return

    pca = PCA(n_components=50)
    pca.fit(images)

    # Create directories if they don't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Save cumulative variance plot
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure()
    plt.plot(cumulative_variance)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs. Number of Principal Components')
    plt.savefig('figures/cumulative_variance.png')

    # Save leading eigenfaces
    for i in range(10):
        eigenface = pca.components[:, i].reshape(112, 92)
        plt.figure()
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {i+1}')
        plt.savefig(f'figures/eigenface_{i+1}.png')

    # Reconstruct the first image
    image_idx = 0
    original_image = images[image_idx]
    transformed_image = pca.transform([original_image])
    reconstructed_image = pca.inverse_transform(transformed_image).flatten()

    # Calculate MSE and MAE
    mse = mean_squared_error(original_image, reconstructed_image)
    mae = mean_absolute_error(original_image, reconstructed_image)
    print(f'MSE: {mse:.2f}, MAE: {mae:.2f}')

    # Save the results
    with open('results/reconstruction_results.txt', 'w') as f:
        f.write(f'MSE: {mse:.2f}\n')
        f.write(f'MAE: {mae:.2f}\n')

    # Plot the original and reconstructed images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image.reshape(112, 92), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image.reshape(112, 92), cmap='gray')
    plt.savefig('figures/reconstructed_image.png')
    plt.show()