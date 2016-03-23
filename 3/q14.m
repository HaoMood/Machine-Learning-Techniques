clear, clc, close all;
load train.dat;
X = train(:, 1: end - 1);
y = train(:, end);
load test.dat;
X_test = test(:, 1: end - 1);
y_test = test(:, end);
[m, n] = size(X);
m_test = size(X_test, 1);

totInNodes = 0;
[ tree totInNodes ] = decisionTree(X, y, totInNodes);

hat_y = predict( X, tree );

sum( hat_y ~= y ) / m