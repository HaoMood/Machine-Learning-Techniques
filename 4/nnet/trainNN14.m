function [W1, b1, W2, b2, W3, b3] = trainNN(X, y,  M, alpha, r)
    [m, n] = size(X);
    T = 50000;
    r = 0.1;
    alpha = 0.01;

    n1 = n;
    n2 = 8;
    n3 = 3;
    n4 = 1;

    W1 = 2 * r * rand(n2, n1) - r;
    b1 = 2 * r * rand(n2, 1) - r;
    W2 = 2 * r * rand(n3, n2) - r;
    b2 = 2 * r * rand(n3, 1) - r;
    W3 = 2 * r * rand(n4, n3) - r;
    b3 = 2 * r * rand(n4, 1) - r;

    for t = 1: T
        i = randi(m);
        a1 = X(i, :)';
        s2 = W1 * a1 + b1;
        a2 = tanh(s2);
        s3 = W2 * a2 + b2;
        a3 = tanh(s3);
        s4 = W3 * a3 + b3;
        a4 = tanh(s4);

        delta4 = -2 * (y(i) - a4) * (1 - tanh(s4).^2);
        DW3 = delta4 * a3';
        Db3 = delta4;
        delta3 = W3' * delta4 .* (1 - tanh(s3).^2);
        DW2 = delta3 * a2';
        Db2 = delta3;
        delta2 = W2' * delta3 .* (1 - tanh(s2).^2);
        DW1 = delta2 * a1';
        Db1 = delta2;

        W3 = W3 - alpha * DW3;
        W2 = W2 - alpha * DW2;
        W1 = W1 - alpha * DW1;
        b3 = b3 - alpha * Db3;
        b2 = b2 - alpha * Db2;
        b1 = b1 - alpha * Db1;
    end