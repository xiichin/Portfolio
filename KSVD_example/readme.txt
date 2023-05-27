Haziq Saharuddin
a1788333@student.adelaide.edu.au 


KSVD_example README
July 09, 2021

Parameter settings:
--------------------
S = 5;      % sparsity
N = 20;     % length of each observation, i.e. time series vector, based on the journal 20*50
L = 100;    % number of observations
K = 50;     % dictionary size
maxIts = 120;   %change 220
snr = 30;   % dB
lambda_list = [0.1, 0.3, 0.5, 0.9];
lambda_count = length(lambda_list);
* 0.3 > lambda > 0.35 . Try change lambda for 1, 0.5 and 0.01
  Using lambda = 0.3 returned an average value almost equal to the targeted
  sparsity while reducing the lambda i.e 0.2 will over-penalised the
  sparsity and vice-versa. Reducing it leads to poorer sparsity and better 
  representation. i.e. Lower reconstruction error and higher SNR
  lambda value increased with increased of number of observation,L 
  i.e. L = 200, lambda = 0.4 . L = 100, lambda = 0.3


KSVD_example.m:
---------------------

1. KSVD_example.m contains an implementation of K-SVD algorithm using the OMP as the matching pursuit
2. The code generates an artificial data to train a dictionary, D then solve the give artificial data.
3. The second part of the code introduces an unseen artificial data. Using the trained dictionary, D,
   to solve the unseen problem.
4. The experiment seeks to find:
	- The average value of the coefficient
	- Training time taken to train the dictionary and solve the generated and unseen problem
	- Evaluate the output SNR error of the generated and unseen problem



KSVD_with_FISTA.m:
--------------------

1. Using the same setting from the KSVD_example.m, the OMP now been replaced with FISTA as the sparse coding method
2. Using the FISTA, the code required a lambda parameter needed to varied. 
   To balance between the sparsity and good approximation
3. The experiment seeks to find:
	- The average value of the coefficient
	- Training time taken to train the dictionary and solve the generated and unseen problem
	- Evaluate the output SNR error of the generated and unseen problem

Conclusion
-----------
1. OMP performs better as it able to calculate the average of the coefficient equal to sparsity parameter
   and faster than FISTA 


Reference
---------