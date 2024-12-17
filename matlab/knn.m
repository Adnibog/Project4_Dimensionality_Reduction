function [y_pred] = knn(X_train, y_train, X_test, k)
    % X_train: Training data matrix (N x p)
    % y_train: Labels vector (1 x N)
    % X_test: Test data matrix (M x p)
    % k: The number of nearest neighbors to consider.
    
    M = size(X_test, 1);
    y_pred = zeros(M, 1);

    for i = 1:M
        % Compute distances between the test point and all training points
        distances = sqrt(sum((X_train - X_test(i, :)).^2, 2));
        [~, idx] = sort(distances);
        
        % Get the majority class of the k nearest neighbors
        nearest_classes = y_train(idx(1:k));
        y_pred(i) = mode(nearest_classes);
    end
end

