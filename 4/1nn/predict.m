function h = predict(X, y, X_test);
    [m, n] = size(X);
    m_test = size(X_test, 1);

    h = zeros(m_test, 1);

    for i = 1: m_test
        dist = sum((X' - X_test(i, :)' * ones(1, m)).^2);
        [~, C] = min(dist);
        h(i) = y(C);
    end