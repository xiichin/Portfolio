function [w, infos] = fista(problem, options, maxIts)
% Fast iterative soft (shrinkage)-thresholding algorithm (FISTA) for LASSO problem.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of SparseGDLibrary.
%
% Created by H.Kasai on Apr. 18, 2017
% Modified by H.Kasai on Apr. 24, 2017


    % set dimensions and samples
    d = problem.dim();
    if isfield(problem, 'n')
        n = problem.samples(); 
    else 
        n = 1;
    end
%     if isfield(problem, 'Lap')
%         Lap = problem.Lap();
%     else
%         Lap = eye(d);
%     end

    
    % extract options
    if ~isfield(options, 'tol_sol_optgap')
        tol_sol_optgap = 1.0e-12;
    else
        tol_sol_optgap = options.tol_sol_optgap;
    end      
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end      
    
    if ~isfield(options, 'tol_gnorm')
        tol_gnorm = 1.0e-12;
    else
        tol_gnorm = options.tol_gnorm;
    end    
    
    if ~isfield(options, 'max_iter')
        max_iter = 100;      %default 100. Change to maxIts parameter from main.m 
    else
        max_iter = options.max_iter;
    end 
    
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end   
    
    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end    
    
    if ~isfield(options, 'solution')
        solution = -Inf;
    else
        solution = options.solution;
    end      
    
    if ~isfield(options, 'store_w')
        store_w = false;
    else
        store_w = options.store_w;
    end
    
    % Lipschitz constant of the gradient of f
    if ~isfield(options, 'L')
        if isfield(problem, 'L')
            L = problem.L();
        else
            L = 1;
        end
    else
        L = options.L;
    end     
    
    % initialise
    iter = 0;
    Linv = 1/L;
    w_prev = w;
    y_prev = w;
    t_prev = 1;
    
    % store first infos
    clear infos;
    infos.iter = iter;
    infos.time = 0;    
    infos.grad_calc_count = 0;    
    f_val = problem.cost(w);
    infos.cost = f_val;     
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    sol_optgap = norm(w - solution);
    infos.sol_optgap = sol_optgap;        
    grad = problem.full_grad(w);
    gnorm = norm(grad);
    infos.gnorm = gnorm;
    if isfield(problem, 'reg')
        infos.reg = problem.reg(w);   
    end    
    if store_w
        infos.w = w;       
    end
    
    % set start time
    start_time = tic();  
    
    % print info
%     if verbose
%         fprintf('FISTA: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e, solution optgap = %.4e\n', iter, f_val, gnorm, optgap, sol_optgap);
%     end      

    % main loop
    %while (optgap > tol_optgap) && (sol_optgap > tol_sol_optgap) && (gnorm > tol_gnorm) && (iter < max_iter)        
    while(iter < maxIts)
        % calculate gradient
        grad_y_old = problem.full_grad(y_prev);
        u = y_prev - Linv * grad_y_old;
        w = problem.prox(u, Linv);
      
        % calculate gradient
        grad = problem.full_grad(w);

        % update iter        
        iter = iter + 1;
        % calculate error
        f_val = problem.cost(w);
        optgap = f_val - f_opt;  
        sol_optgap = norm(w - solution);
        % calculate norm of gradient
        gnorm = norm(grad);
        
        % measure elapsed time
        elapsed_time = toc(start_time);        

        % store infoa
        infos.iter = [infos.iter iter];
        infos.time = [infos.time elapsed_time];        
        infos.grad_calc_count = [infos.grad_calc_count iter*n];      
        infos.optgap = [infos.optgap optgap]; 
        infos.sol_optgap = [infos.sol_optgap sol_optgap];     
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm]; 
        if isfield(problem, 'reg')
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end        
        if store_w
            infos.w = [infos.w w];         
        end        
       
        % print info
%         if verbose
%             fprintf('FISTA: Iter = %03d, cost = %.24e, gnorm = %.4e, optgap = %.4e, solution optgap = %.4e\n', iter, f_val, gnorm, optgap, sol_optgap);
%         end  
        
        % update paramters
        t = 0.5*(1 + sqrt(1 + 4 * t_prev^2));
        y = w + (t_prev - 1)/t * (w - w_prev);
        
        % store parameters
        w_prev = w;
        t_prev = t;
        y_prev = y;           
    end
    
%     if gnorm < tol_gnorm
%         fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', tol_gnorm);
%     elseif optgap < tol_optgap
%         fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);          
%     elseif sol_optgap < tol_sol_optgap
%         fprintf('Solution optimality gap tolerance reached: tol_sol_optgap = %g\n', tol_sol_optgap);        
    if iter == max_iter
%         fprintf('Max iter reached: max_iter = %g\n', max_iter);
    end    
    
end
