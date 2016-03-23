function [W1, b1, W2, b2] = trainNN(X, y,  M, alpha, r)
    [m, n] = size(X);
    T = 50000;

    n1 = n;
    n2 = M;
    n3 = 1;

    W1 = 2 * r * rand(n2, n1) - r;
    b1 = 2 * r * rand(n2, 1) - r;
    W2 = 2 * r * rand(n3, n2) - r;
    b2 = 2 * r * rand(n3, 1) - r;

    for t = 1: T
        i = randi(m);
        a1 = X(i, :)';
        s2 = W1 * a1 + b1;
        a2 = tanh(s2);
        s3 = W2 * a2 + b2;
        a3 = tanh(s3);

        delta3 = -2 * (y(i) - a3) * (1 - tanh(s3).^2);
        DW2 = delta3 * a2';
        Db2 = delta3;
        delta2 = W2' * delta3 .* (1 - tanh(s2).^2);
        DW1 = delta2 * a1';
        Db1 = delta2;

        W2 = W2 - alpha * DW2;
        W1 = W1 - alpha * DW1;
        b2 = b2 - alpha * Db2;
        b1 = b1 - alpha * Db1;
    end