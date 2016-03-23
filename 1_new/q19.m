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
 
 best_acc = 0;
 best_gamma = -1;

for gam = [1 1e1 1e2 1e3 1e4]
    cmd = ['-c 0.1 -g ' num2str(gam)];
    model = libsvmtrain(y, X, cmd);
    [~, acc, ~] = libsvmpredict(y_test, X_test, model);
    disp(acc);
    if best_acc < acc(1)
        best_acc = acc(1);
        best_gamma = gam;
    end
end

best_gamma