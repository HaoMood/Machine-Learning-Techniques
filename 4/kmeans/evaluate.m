function J = evaluate(X, c, mu)
    [m, n] = size(X);
    K = size(mu, 2);
    J = 0;
    for k = 1: K
        Xk = X(c == k, :)';
        mk = size(Xk, 2);
        J = J + sum(sum((Xk - mu(:, k) * ones(1, mk)).^2));
    end
    J = J / m;