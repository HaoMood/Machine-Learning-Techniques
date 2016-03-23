function J_out = testNN(X, y, W1, b1, W2, b2)
    [m, n] = size(X);

    a1 = X';
    s2 = W1 * a1 + b1 * ones(1, m);
    a2 = tanh(s2);
    s3 = W2 * a2 + b2 * ones(1, m);
    a3 = tanh(s3);
    h = sign(a3)';

    J_out = sum(h ~= y) / m;