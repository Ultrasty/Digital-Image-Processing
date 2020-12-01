function s = ifrdescp(z, nd)
%IFRDESCP Computes inverse Fourier descriptors.
%   S = IFRDESCP(Z, ND) computes the inverse Fourier descriptors of
%   of Z, which is a sequence of Fourier descriptor obtained, for
%   example, by using function FRDESCP.  ND is the number of
%   descriptors used in computing the inverse; ND must be an even
%   integer no greater than length(Z).  If ND is omitted, it defaults
%   to length(Z).  The output, S, is a length(Z)-by-2 matrix containing
%   the coordinates of a closed boundary.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.7 $  $Date: 2005/01/18 13:44:34 $

% Preliminaries.
np = length(z);
% Check inputs.
if nargin == 1 | nd > np 
   nd = np; 
end

% Create an alternating sequence of 1s and -1s for use in centering
% the transform.
x = 0:(np - 1);
m = ((-1) .^ x)';

% Use only nd descriptors in the inverse.  Since the 
% descriptors are centered, (np - nd)/2 terms from each end of 
% the sequence are set to 0.  
d = round((np - nd)/2); % Round in case nd is odd.
z(1:d) = 0;
z(np - d + 1:np) = 0;
% Compute the inverse and convert back to coordinates.
zz = ifft(z);
s(:, 1) = real(zz);
s(:, 2) = imag(zz);
% Multiply by alternating 1 and -1s to undo the earlier 
% centering.
s(:, 1) = m.*s(:, 1);
s(:, 2) = m.*s(:, 2);
