function [c, mu] = kmeans(X, K)
    [m, n] = size(X);
    pm = randperm(m);
    mu = X(pm(1: K), :)';
    c_old = ones(m, 1);
    c = zeros(m, 1);
    dist = zeros(K, m);

    while ~isequal(c_old, c)
        c_old = c;
        for k = 1: K
            dist(k, :) = sum((X' - mu(:, k) * ones(1, m)).^2);
        end
        [~, c] = min(dist);

        for k = 1: K
            mu(:, k) = sum(ones(n, 1) * (c == k) .* X', 2) / sum(c == k);
        end
    end