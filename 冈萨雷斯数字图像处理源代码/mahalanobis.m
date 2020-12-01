function d = mahalanobis(varargin)
%MAHALANOBIS Computes the Mahalanobis distance.
%   D = MAHALANOBIS(Y, X) computes the Mahalanobis distance between
%   each vector in Y to the mean (centroid) of the vectors in X, and
%   outputs the result in vector D, whose length is size(Y, 1).  The
%   vectors in X and Y are assumed to be organized as rows.  The
%   input data can be real or complex. The outputs are real
%   quantities.
%
%   D = MAHALANOBIS(Y, CX, MX) computes the Mahalanobis distance
%   between each vector in Y and the given mean vector, MX. The
%   results are output in vector D, whose length is size(Y, 1).  The
%   vectors in Y are assumed to be organized as the rows of this
%   array. The input data can be real or complex. The outputs are
%   real quantities. In addition to the mean vector MX, the
%   covariance matrix CX of a population of vectors X also must be
%   provided. Use function COVMATRIX (Section 11.5) to compute MX and
%   CX.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.6 $  $Date: 2005/01/18 13:44:47 $

% Reference: Acklam, P. J. [2002]. "MATLAB Array Manipulation Tips
% and Tricks." Available at
%     home.online.no/~pjacklam/matlab/doc/mtt/index.html 
% or at
%     www.prenhall.com/gonzalezwoodseddins

param = varargin; % Keep in mind that param is a cell array.
Y = param{1};
ny = size(Y, 1); % Number of vectors in Y.

if length(param) == 2
   X = param{2};
   % Compute the mean vector and covariance matrix of the vectors
   % in X.
   [Cx, mx] = covmatrix(X);
elseif length(param) == 3 % Cov. matrix and mean vector provided.
   Cx = param{2};
   mx = param{3};
else 
   error('Wrong number of inputs.')
end
mx = mx(:)'; % Make sure that mx is a row vector.

% Subtract the mean vector from each vector in Y.
Yc = Y - mx(ones(ny, 1), :);	

% Compute the Mahalanobis distances.
d = real(sum(Yc/Cx.*conj(Yc), 2));
