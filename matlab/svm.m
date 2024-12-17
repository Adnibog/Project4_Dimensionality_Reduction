function [y_pred] = svm(X_train, y_train, X_test)
    % X_train: Training data matrix of size (N x p)
    % y_train: Labels vector (1 x N) for training data
    % X_test: Test data matrix of size (M x p)
    
    SVMModel = fitcsvm(X_train, y_train);  % Train the SVM model
    y_pred = predict(SVMModel, X_test);   % Predict labels for the test set
end
