function [opt_theta opt_s opt_j] = decisionStump(X, y, u)
opt_theta = -1e3;
opt_s = -1e3;
opt_j = -1;
opt_J = 1e6;
[m, n] = size(X);
for j = 1: n
    xj_sort = sort(unique(X(:, j)))';
    Theta = [-1e3 (xj_sort(2: end) + xj_sort(1: end-1)) / 2];
    s_j = ones(size(Theta, 2), 1);

    hat_y = sign(X(:, j) * ones(1, size(Theta, 2)) - ones(m, 1) * Theta);
    J = sum((y * ones(1, size(Theta, 2)) ~= hat_y) .* (u * ones(1, size(Theta, 2)))) / sum(u);

    s_j(J > 0.5) = -1;
    J(J > 0.5) = 1 - J(J > 0.5);

    min_idx = find(J == min(J));

    if min(J) == opt_J
        opt_j = [opt_j j * ones(1, size(min_idx, 2))];
        opt_theta = [opt_theta Theta(min_idx)];
        opt_s = [opt_s s_j(min_idx)];
    end

    if min(J) < opt_J
        opt_J = min(J);
        opt_j = j;
        opt_theta = Theta(min_idx);
        opt_s = s_j(min_idx);
    end
end

randIdx = randi(size(opt_j, 2));

opt_j = opt_j(randIdx);
opt_theta = opt_theta(randIdx);
opt_s = opt_s(randIdx);

% opt_J