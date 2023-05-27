function [idx, x, xFull, idxArray, xArray] = OMP(K,A,s,errThres)
%OMP Othorgonal Matching Pursuit (OMP)
%
%   Type of greedy pursuit that identifies the largest elements in the
%   coefficient vector, and applies least square processing for each
%   iteration. Iterating continues until a stopping criterion is met.
%   REFERENCE: J. A. Tropp and A. C. Gilbert, "Signal Recovery From Random
%   Measurements Via Orthogonal Matching Pursuit," in IEEE Transactions on
%   Information Theory, vol. 53, no. 12, pp. 4655-4666, Dec. 2007.
%
%   INPUTS:
%       K:          Maximum sparsity level (maximum number of collected atoms)
%       A:          Dictionary
%       s:          Received signal
%     errThres:   Residual threshold - stopping criterion parameter
%
%   OUTPUTS:
%       idx:        Final index set - location of atoms in dictionary
%       x:          Reconstructed version of the original signal - coefficient
%       xFull:      Full version of x
%       idxArray:	A matrix storing index sets `I' for all iterations
%       xArray:     A matrix storing coefficient `x' for all iterations
%
%   USAGE:
%       [idx, x, xFull, idxArray, xArray] = OMP(K,A,s,errThres)
%
%   WRITTEN BY: Ngoc Hung Nguyen and Kultuyil Dogancay, 2016 (University of South Australia)
%
%   CLASSIFICATION: UNCLASSIFIED
%
%   See Also: CoSaMP, gOMP, rOMP, stOMP, subSpP, SWOMP
%
%   NCTR TOOLBOX VERSION 3 - LIBRARY FUNCTION
%   DEFENCE SCIENCE AND TECHNOLOGY GROUP, AUSTRALIA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
narginchk(1,4)


% Set residual
r=s;

%Set index set to "empty"
idx = [];

%
loop=1;

while (length(idx)<K && norm(r)>errThres)
    
    %Match filter
    c = A'*r;
    
    %Identify
    c_abs=abs(c);
    [~, ix] = sort(c_abs, 'descend');
    j = ix(1);
    
    %Set Union
    idxnew = union(idx,j);
    
    if length(idxnew)==length(idx)
        break;      % stop iteration if no new atom is added
    else
        idx = idxnew;
        %Projection (Least square)
        ASubidx = A(:,idx);
        x = lscov(ASubidx,s);
%         size(x)
        
        %Update residual
        r = s-ASubidx*x;
        rVector(loop)=norm(r);
%         norm(r)
        
        %Save cofficient for plotting
        xArray(length(x),loop)=0;
        xArray(:,loop)=x;
        
        %Save Index for plotting
        idxArray(length(idx),loop)=0;
        idxArray(:,loop)=idx;
        
        loop=loop+1;
        
        
    end
end

xFull=zeros(size(A,2),1);
xFull(idx,1)=x;
