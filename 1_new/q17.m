clear, clc
load train.txt;
X = train(:, 2: end);
y = train(:, 1);
[m, n] = size(X);
% load test.txt;
% X_test = test(:, 2: end);
% y_test = test(:, 1);
% m_test = size(X_test, 1);

max_alpha = 0;

for dig = 0: 2: 8
    y_train = ones(m, 1);
    y_train(y==dig) = -1;
    model = libsvmtrain(y_train, X, '-t 1 -g 1 -r 1 -d 2 -c 0.01');
    alpha = zeros(m, 1);
    alpha(model.sv_indices) = y_train(model.sv_indices) .* model.sv_coef;
    disp(alpha');
    if max_alpha < norm(alpha, 1)
        max_alpha = norm(alpha, 1);
    end
end

max_alpha