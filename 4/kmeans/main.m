clear, clc, close all
rng(0);
load train.dat;
X = train;
clear train
[m, n] = size(X);

totRounds = 500;
% K = 2;
K = 10;
J_in = zeros(totRounds, 1);

for rounds = 1: totRounds
    [c, mu] = kmeans(X, K);
    J_in(rounds) = evaluate(X, c, mu);
end
sum(J_in) / totRounds