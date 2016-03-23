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
M = [1 6 11 16 21];

J_out = zeros(totRounds, 5);
for k = 1: 5
    for rounds = 1: totRounds
        [W1, b1, W2, b2] = trainNN(X, y, M(k));
        J_out(rounds, k) = testNN(X_test, y_test, W1, b1, W2, b2, M(k));

        if mod(rounds, 10) == 0
            J_in = testNN(X, y, W1, b1, W2, b2, M(k));
            fprintf('k = %d, rounds = %d, J_in = %f, J_out = %f\n', k, rounds, J_in, J_out(rounds, k));
        end
    end
end
sum(J_out) / totRounds