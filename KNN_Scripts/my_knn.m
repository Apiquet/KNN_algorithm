function [ y_est ] =  my_knn(X_train,  y_train, X_test, params)
%MY_KNN Implementation of the k-nearest neighbor algorithm
%   for classification.
%
%   input -----------------------------------------------------------------
%   
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {1,2} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o params : struct array containing the parameters of the KNN (k,
%                  d_type and eventually the parameters for the Gower
%                  similarity measure)
%
%   output ----------------------------------------------------------------
%
%       o y_est   : (1 x M_test), a vector with estimated labels y \in {1,2} 
%                   corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
[N, M_test]  = size(X_test);
[~, M_train] = size(X_train);
y_est        = zeros(1, M_test);

Matrix_dist = zeros(M_test,M_train);

for i = 1:M_test
    for j = 1:M_train
        Matrix_dist(i,j) = my_distance(X_test(:,i),X_train(:,j),params);
    end
end

for i = 1:M_test
    [sorted_line,k_min_index] = sort(Matrix_dist(i,:));
    sorted_line = sorted_line(1:params.k);
    k_min_index = k_min_index(1:params.k);
    labels=zeros(1,params.k);
    for j = 1:params.k
        labels(j) = y_train(k_min_index(j));
    end
    [value,freq] = mode(labels(:));
    y_est(i) = value;
end



end