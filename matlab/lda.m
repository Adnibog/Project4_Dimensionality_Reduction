function [Y, coeff] = lda(X, y, dims)
    % X: Data matrix of size (N x p), where N is the number of samples, p is the number of features.
    % y: The labels vector (1 x N), representing class labels for each sample.
    % dims: The number of dimensions to reduce to.
    
    % Perform LDA
    [coeff, ~, ~] = lda(X, y);
    Y = X * coeff(:, 1:dims);  % Project data onto the top 'dims' LDA components
    
    figure;
    scatter(Y(:,1), Y(:,2), 30, 'filled');
    title('LDA Visualization');
end

