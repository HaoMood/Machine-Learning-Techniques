function q2()
    clear, clc;
    X = [1, 0; 0 1; 0, -1; -1, 0; 0, 2; 0, -2; -2 0];
    y = [-1; -1; -1; 1; 1; 1; 1];
    [m, n] = size(X);
    show(X, y);

    Z = zeros(m, n);
    Z(:, 1) = X(:, 2).^2 - 2 * X(:, 1) + 3;
    Z(:, 2) = X(:, 1).^2 - 2 * X(:, 2) - 3;
    show(Z, y)


function show(X, y)
    y1_idx = y == 1;
    figure, grid on, hold on;
    plot(X(y1_idx, 1), X(y1_idx, 2), 'ro', 'MarkerSize', 10);
    plot(X(~y1_idx, 1), X(~y1_idx, 2), 'bx', 'MarkerSize', 10);
