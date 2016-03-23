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

tot_err_train = 0;
tot_err_test = 0;
for r = 1: rounds
    hat_y_train = zeros( m, T );
    hat_y_test = zeros( m_test, T );

    % build a RF
    for t = 1: T
        bs = randi( m, m, 1 );
        tree = giniStump( X(bs, :), y(bs) );

        hat_y_train(:, t) = predictStump( X, tree );
        hat_y_test(:, t) = predictStump( X_test, tree );

    end
    tot_err_train = tot_err_train + sum( sign( sum( hat_y_train, 2 ) ) ~= y ) / m;
    tot_err_test = tot_err_test + sum( sign( sum( hat_y_test, 2 ) ) ~= y_test ) / m_test;
    disp(r);
end

tot_err_train / rounds
tot_err_test / rounds
