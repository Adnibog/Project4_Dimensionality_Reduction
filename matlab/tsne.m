function [Y] = tsne(X, dims)
    % X: Data matrix of size (N x p), where N is the number of samples, p is the number of features.
    % dims: The number of dimensions to reduce to (e.g., 2 for visualization).
    
    Y = tsne(X, 'NumDimensions', dims);  % Reduce dimensions
    figure;
    scatter(Y(:,1), Y(:,2), 30, 'filled'); % Scatter plot in 2D
    title('t-SNE Visualization');
end

