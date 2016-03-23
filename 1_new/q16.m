clear, clc
load train.txt;
X = train(:, 2: end);
y = train(:, 1);
[m, n] = size(X);
load test.txt;
X_test = test(:, 2: end);
y_test = test(:, 1);
m_test = size(X_test, 1);

max_acc = 0;
max_acc_dig = 0;

for dig = 0: 2: 8
    y_train = ones(m, 1);
    y_train(y==dig) = -1;
    model = libsvmtrain(y_train, X, '-t 1 -g 1 -r 1 -d 2 -c 0.01');
    [~, acc, ~] = libsvmpredict(y_train, X, model);
    disp(acc)
    disp(dig)
    if max_acc < acc
        max_acc = acc;
        max_acc_dig = dig;
    end
end

dig