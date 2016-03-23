clear, clc
load train.txt;
X = train(:, 2: end);
y = train(:, 1);
[m, n] = size(X);
load test.txt;
X_test = test(:, 2: end);
y_test = test(:, 1);
m_test = size(X_test, 1);

y(y==0) = -1;
y(y>=1) = 1;
model = libsvmtrain(y, X, '-t 0 -c 0.01');
w = model.SVs' * model.sv_coef;
b = -model.rho;
if model.Label(1) == -1
  fprintf('in -1');
  w = -w;
  b = -b;
end
norm(w)