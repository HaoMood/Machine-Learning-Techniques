clear, clc
load train.txt;
X = train(:, 2: end);
y = train(:, 1);
[m, n] = size(X);
y(y==0) = -1;
y(y>=1) = 1;
% load test.txt;
% X_test = test(:, 2: end);
% y_test = test(:, 1);
% m_test = size(X_test, 1);
% y_test(y_test==0) = -1;
% y_test(y_test>=1) = 1;
 
rng(0);
gamma_stat = zeros(5, 1);

for t = 1: 100
    best_acc = 0;
    best_gamma = -1;

    p = randperm(m);
    X_train = X(p(1: 1e3), :);
    y_train = y(p(1: 1e3));
    X_val = X(p(1e3 + 1: end), :);
    y_val = y(p(1e3 + 1: end));

    for gamma = [1 1e1 1e2 1e3 1e4]
        cmd = ['-c 0.1 -g ' num2str(gamma)];
        model = libsvmtrain(y_train, X_train, cmd);
        [~, acc, ~] = libsvmpredict(y_val, X_val, model);
        disp(acc);
        if best_acc < acc(1)
            best_acc = acc(1);
            best_gamma = gamma;
        end
    end

    gamma_stat(int8(log10(best_gamma)+1)) = gamma_stat(int8(log10(best_gamma)+1)) + 1;
end

gamma_stat