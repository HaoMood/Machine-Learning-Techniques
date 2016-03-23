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

rng(0);
totRounds = 500;

J_out = zeros(totRounds, 1);
for rounds = 1: totRounds
    [W1, b1, W2, b2, W3, b3] = trainNN14(X, y);
    J_out(rounds) = testNN14(X_test, y_test, W1, b1, W2, b2, W3, b3);

    if mod(rounds, 50) == 0
        J_in = testNN14(X, y, W1, b1, W2, b2, W3, b3);
        fprintf('rounds = %d, J_in = %f, J_out = %f\n', rounds, J_in, J_out(rounds));
    end
end
sum(J_out) / totRounds