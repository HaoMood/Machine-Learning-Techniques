clear, clc, close all;
load lssvm.dat;
X = lssvm(1: 400, 1: end - 1);
y = lssvm(1: 400, end);
X_test = lssvm(401: end, 1: end - 1);
y_test = lssvm(401: end, end);
[m, n] = size(X);
m_test = size(X_test, 1);

for gam = [32 2 0.125]
    K = zeros(m, m);
    for i = 1: m
        xi = X(i, :)';
        K(i, :) = exp(-gam * sum((X' - xi * ones(1, m)).^2));
    end
    
    K_test = zeros(m_test, m);
    for i = 1: m_test
        xi = X_test(i, :)';
        K_test(i, :) = exp(-gam * sum((X' - xi * ones(1, m)).^2));
    end

    for lambda = [0.001 1 1000]
        beta = inv(lambda * eye(m, m) + K) * y;

        hx = sign(K * beta);
        hx_test = sign(K_test * beta);

        J_in = sum(hx ~= y) / m;
        J_out = sum(hx_test ~= y_test) / m_test;
        fprintf('gamma = %f\t lambda = %f\t J_in = %f\t J_out = %f\n', gam, lambda, J_in, J_out);
    end
end