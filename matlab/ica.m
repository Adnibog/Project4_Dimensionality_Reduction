function [S] = ica(X, dims)
    % X: Data matrix of size (N x p), where N is the number of samples, p is the number of features.
    % dims: The number of dimensions to reduce to.
    
    % Perform ICA using the 'fastica' method (requires the FastICA toolbox)
    [S, ~, ~] = fastica(X', 'numOfComponents', dims);
    
    % Plot the independent components
    figure;
    scatter(S(1, :), S(2, :), 30, 'filled');
    title('ICA Visualization');
end
