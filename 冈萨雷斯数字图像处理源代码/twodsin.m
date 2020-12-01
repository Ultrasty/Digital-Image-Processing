function [rt, f, g] = twodsin(A, u0, v0, M, N)
%TWODSIN Compare for-loops vs. vectorization.
%   The comparison is based on implementing the function f(x, y) =
%   Asin(u0x + v0y) for x = 0, 1, 2,..., M - 1 and y = 0, 1, 2,...,
%   N - 1. The inputs to the function are M and N and the constants
%   in the function. 
%
%   Sample M-file used in Chapter 2.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.3 $  $Date: 2003/10/13 00:53:47 $

% First implement using for loops.
 
tic   % Start timing.

for r = 1:M
   u0x = u0*(r - 1);
   for c = 1:N
      v0y = v0*(c - 1);
      f(r, c) = A*sin(u0x + v0y);
   end
end

t1 = toc;   % End timing.

%	Now implement using vectorization.  Call the image g.

tic   % Start timing.

r = 0:M - 1;
c = 0:N - 1;
[C, R] = meshgrid(c, r);
g = A*sin(u0*R + v0*C);

t2 = toc;   % End timing.
 
%	Compute the ratio of the two times.

rt = t1/(t2 + eps); % Use eps in case t2 is close to 0
