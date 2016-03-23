clear, clc, close all;
load train.dat;
X = train(:, 1: end - 1);
y = train(:, end);
load test.dat;
X_test = test(:, 1: end - 1);
y_test = test(:, end);
[m, n] = size(X);
m_test = size(X_test, 1);

u = ones(m, 1) / m;
T = 300;
rng(0);

alpha = zeros(T, 1);
hx_train = zeros(m, T);
hx_test = zeros(m_test, T);

for t = 1: T
    % sum(u)

    [theta s j] = decisionStump(X, y, u);

    htx = s * sign(X(:, j) - theta);
    epsilon = sum(u .* (htx ~= y)) / sum(u);

    % epsilon

    rho = sqrt((1 - epsilon) / epsilon);
    alpha(t) = log(rho);

    u(htx ~= y) = u(htx ~= y) * rho;
    u(htx == y) = u(htx == y) / rho;

    hx_train(:, t) = s * sign(X(:, j) - theta);
    hx_test(:, t) = s * sign(X_test(:, j) - theta);

    sum(hx_test(:, t) ~= y_test) / m_test
end

hat_y_train = sign(hx_train * alpha);
J_in = sum(y ~= hat_y_train) / m;

hat_y_test = sign(hx_test * alpha);
J_out = sum(y_test ~= hat_y_test) / m_test;

fprintf('J_in = %f, J_out = %f\n', J_in, J_out);