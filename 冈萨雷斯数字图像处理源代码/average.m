function av = average(A)
%AVERAGE Computes the average value of an array.
%   AV = AVERAGE(A) computes the average value of input array, A,
%   which must be a 1-D or 2-D array.  

%   Sample M-file used in Chapter 2.
%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.2 $  $Date: 2003/03/27 21:23:21 $

% Check the validity of the input. (Keep in mind that
% a 1-D array is a special case of a 2-D array.)
if ndims(A) > 2
   error('The dimensions of the input cannot exceed 2.')
end

% Compute the average
av = sum(A(:))/length(A(:));
