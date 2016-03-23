clear, clc, close all;
load train.dat;
X = train(:, 1: end - 1);
y = train(:, end);
load test.dat;
X_test = test(:, 1: end - 1);
y_test = test(:, end);
[m, n] = size(X);
m_test = size(X_test, 1);

rng(0);
rounds = 100;
T = 300;

tot_err = 0;
for r = 1: rounds
    hat_y_train = zeros( m, T );
    hat_y_test = zeros( m_test, T );

    % build a RF
    for t = 1: T
        bs = randi( m, m, 1 );
        [ tree, ~ ] = decisionTree( X(bs, :), y(bs), 0 );

        hat_y_train(:, t) = predict( X, tree );
        hat_y_test(:, t) = predict( X_test, tree );

    end
    tot_err = tot_err + sum( sum( hat_y_train ~= y * ones( 1, T ) ) ) / m / T;
    disp(r);
end

tot_err / rounds