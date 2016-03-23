clear, clc, close all
load train.dat;
X = train(:, 1: end - 1);
y = train(:, end);
load test.dat;
X_test = test(:, 1: end - 1);
y_test = test(:, end);
[m, n] = size(X);
m_test = size(X_test, 1);
clear train test

h = predict(X, y, X);
J_in = sum(h ~= y) / m
h_test = predict(X, y, X_test);
J_out = sum(h_test ~= y_test) / m_test