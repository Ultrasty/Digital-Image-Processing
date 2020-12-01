function v = gmean(A)
%GMEAN Geometric mean of columns.
%   V = GMEAN(A) computes the geometric mean of the columns of A.  V
%   is a row vector with size(A,2) elements.
%
%   Sample M-file used in Chapter 3.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.2 $  $Date: 2003/04/16 22:31:47 $

m = size(A, 1);
v = prod(A, 1) .^ (1/m);
