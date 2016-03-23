    clear, clc;
    X = [1, 0; 0 1; 0, -1; -1, 0; 0, 2; 0, -2; -2 0];
    y = [-1; -1; -1; 1; 1; 1; 1];
    [m, n] = size(X);
    model = libsvmtrain(y, X, '-t 1 -d 2 -g 1 -r 1 -c 1e6');
    if model.Label(1) == -1
         fprintf('label(1) is -1\n');
         % model.sv_coef = -model.sv_coef;
         % model.rho = -model.rho;
    end
    w = model.SVs' * model.sv_coef;
    b = -model.rho;
    
    alpha = zeros(m, 1);
    alpha(model.sv_indices) = y(model.sv_indices) .* model.sv_coef;
