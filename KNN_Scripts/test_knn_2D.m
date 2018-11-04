clear; 
close all; 
clc;

dataset_path = '../Data/';
rng(42);
seed = rng;

% Load Dataset
load(strcat(dataset_path,'/2d-concentric-circles.mat'))
[~, M] = size(X);
rand_idx = randperm(M);
X = X(:,rand_idx);
y = y(rand_idx);
y = y + 1;

% Visualize Dataset
options.labels      = y;
options.class_names = {'y = 1','y = 2'};
options.title       = '2D Concentric Circles Dataset';

%Using ML function to plot data
h0 = ml_plot_data(X',options);
axis equal

% Select Training/Testing Ratio
valid_ratio = 0.8;

% Split data into a training dataset that kNN can use to make predictions 
% and a test dataset that we can use to evaluate the accuracy of the model.
[X_train, y_train, X_test, y_test] = split_data(X, y, valid_ratio);

% Select k
params.k = 2; 
params.d_type = 'L2';

% Compute y_estimate from k-NN
y_est =  my_knn(X_train, y_train, X_test, params);

% Compute Accuracy
acc =  my_accuracy(y_test, y_est);

% Visualize Split Dataset
options.labels      = y;
options.class_names = [];
options.title       = sprintf('My kNN, valid-ratio = %1.2f, k= %d, Acc = %1.3f',valid_ratio, params.k, acc);
h2 = ml_plot_data(X',options); hold on;
scatter(X_test(1,y_est == 1),X_test(2,y_est == 1),150,'o','MarkerEdgeColor', [1 0 0]);hold on;
scatter(X_test(1,y_est == 2),X_test(2,y_est == 2),150,'o','MarkerEdgeColor', [0 0 1]);
legend({'$y=1$','$y = 2$','$\hat{y} = 1$','$\hat{y} = 2$'},'Interpreter','latex')
axis equal

% Plot K-NN Decision boundary
knn_options.k      = params.k;
knn_options.d_type = params.d_type;
[~, model]= knn_classifier(X_train, y_train, [], knn_options);

%Using ML function to plot model
f_knn     = @(X_test)knn_classifier(X_test, [], model, []);

% Plot Decision Boundary
c_options         = [];
plot_data_options = [];
c_options.dim_swaped     = true;
c_options.show_misclass  = false;
c_options.title          = sprintf('K(%d)-NN Decision Boundary with valid-ratio: %1.2f',params.k,valid_ratio);
if exist('hc','var') && isvalid(hc), delete(hc);end

%Using ML function to plot model
hc = ml_plot_classifier(f_knn,X',y,c_options,plot_data_options);
axis tight