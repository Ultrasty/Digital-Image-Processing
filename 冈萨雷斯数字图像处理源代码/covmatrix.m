function [C, m] = covmatrix(X)
%COVMATRIX Computes the covariance matrix of a vector population.
%   [C, M] = COVMATRIX(X) computes the covariance matrix C and the
%   mean vector M of a vector population organized as the rows of
%   matrix X. C is of size N-by-N and M is of size N-by-1, where N is
%   the dimension of the vectors (the number of columns of X).

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.4 $  $Date: 2003/05/19 12:09:06 $

[K, n] = size(X);
X = double(X);
if n == 1 % Handle special case.
   C = 0;
   m = X;
else
   % Compute an unbiased estimate of m.
   m = sum(X, 1)/K;
   % Subtract the mean from each row of X.
   X = X - m(ones(K, 1), :);
   % Compute an unbiased estimate of C. Note that the product is
   % X'*X because the vectors are rows of X.	
   C = (X'*X)/(K - 1);
   m = m'; % Convert to a column vector.	 
end
