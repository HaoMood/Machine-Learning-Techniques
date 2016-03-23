clear, clc;
X = [4.48e-4; 5.89e-4; 7.34e-4; 8.59e-4; 1e-3; 1.15e-3; 1.28e-3; 1.42e-3];
Z = X.^2;
[m, n] = size(Z);
y = [2.22; 3.58; 5.44; 7.75; 10.3; 13.53; 16.82; 20.69];
figure, hold on, grid on;
plot(Z, y, 'rx');
xlabel('q_v^2');
ylabel('H_e');

Z = [ones(m, 1) Z];
theta = pinv(Z'* Z) * Z' * y

t = 0: 1e-8: 2.5e-6;
m_test = size(t, 2);
h = [ones(1, m_test); t]' * theta;
plot(t, h);

corrcoef(Z(:, 2), y)

