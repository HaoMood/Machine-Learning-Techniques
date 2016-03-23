clear, clc
load train.txt;
X = train(:, 2: end);
y = train(:, 1);
[m, n] = size(X);
y(y==0) = -1;
y(y>=1) = 1;
load test.txt;
X_test = test(:, 2: end);
y_test = test(:, 1);
m_test = size(X_test, 1);
y_test(y_test==0) = -1;
y_test(y_test>=1) = 1;
 

for C = [0.001 0.01 0.1 1 10]
    cmd = ['-g 100 -c ' num2str(C)];
    model = libsvmtrain(y, X, cmd);
    w = model.SVs' * model.sv_coef;
    b = -model.rho;
    if model.Label(1) == -1
        w = -w;
        b = -b;
    end
    [~, acc, ~] = libsvmpredict(y_test, X_test, model);
    [~, ~, dec_val] = libsvmpredict(y, X, model);
    disp(acc);
    alpha = zeros(m, 1);
    alpha(model.sv_indices) = y(model.sv_indices) .* model.sv_coef;
    s = find(alpha > 0 & alpha < C, 1);
    dist = y(s) * dec_val(s) / norm(w)
end