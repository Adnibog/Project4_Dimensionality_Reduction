function [y_pred] = random_forest(X_train, y_train, X_test, num_trees)
    % X_train: Training data matrix (N x p)
    % y_train: Labels vector (1 x N)
    % X_test: Test data matrix (M x p)
    % num_trees: The number of decision trees in the random forest
    
    RFModel = fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', num_trees);
    y_pred = predict(RFModel, X_test);  % Predict the class labels for the test set
end
