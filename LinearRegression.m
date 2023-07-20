
%Reading data from the csv file
data = readtable("House_Rent_Dataset.csv", "ReadVariableNames", true);
y = data.Rent;
X = data(:, [2, 4, 6, 9, 11]);

%Converting Categorical data to one hot encoding
X.AreaType = grp2idx(X.AreaType);
X.FurnishingStatus = grp2idx(X.FurnishingStatus);

% Plotting Features
subplot(1, 3, 1)
plot(X.Size, y, 'M*', 'Color',	'red')
xlabel('Size')
ylabel('Rent')
title('Size vs Rent')

subplot(1, 3, 2)
plot(X.BHK, y, 'M+', 'Color', 'g')
xlabel("BHK")
ylabel('Rent')
title('BHK vs Rent')

subplot(1, 3, 3)
plot(X.Bathroom, y, 'Mo', 'Color', 'b')
xlabel('Number of Bathrooms')
ylabel('Rent')
title('Number of Bathrooms vs Rent')

X = table2array(X);

%Normalizing X
mean_x = mean(X, 1);
std_x = std(X, 1);
X = (X - mean_x) ./ std_x;
num_train_rows = int64(size(X, 1) * 0.8);

%Normalizing Y
mean_y = mean(y);
std_y = std(y);
y = (y - mean_y) ./ std_y;

y_train = y(1: num_train_rows, :);
y_test = y(num_train_rows + 1:size(y, 1), :);

%Number of features
n = size(X, 2);
%Defining the Linear Regression Code
theta = rand(n + 1, 1);
%Appending 1 infront of each row for X0 = 1
X = cat(2, ones(size(X, 1), 1), X);
X_train = X(1: num_train_rows, :);
X_test = X(num_train_rows + 1:size(X, 1), :);
%Learning rate
learning_rate = 0.01;
lambda = 1;
m = length(y);
iter = 300;
J_history = zeros(iter, 1);
for i = 1:1:iter
    J_history(i) = costFunction(theta, X_train, y_train, lambda);
    theta = grad(theta, X_train, y_train, learning_rate, lambda);
end

figure; hold on;
%plotting loss
plot(1:iter, J_history);
xlabel('Epochs');
ylabel('Loss');
title('Loss over Iter');

cost = costFunction(theta, X_train, y_train, lambda);
fprintf('Cost after 1500 epochs is %0.2f\n', cost);
%Testing the results
testing_loss = costFunction(theta, X_test, y_test, lambda);
fprintf('Cost for testing is: %0.2f\n', testing_loss);

%Testing Prediction
X_custom_pred = [1, 2, 200, 1, 3, 3];
fprintf("Prediction of Rent for House of 2 BHK with size of 500 and 3 bathrooms is: %0.2f\n", predict(theta, X_custom_pred) * std_y + mean_y);
