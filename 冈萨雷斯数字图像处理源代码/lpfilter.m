function H = lpfilter(type, M, N, D0, n)
%LPFILTER Computes frequency domain lowpass filters.
%   H = LPFILTER(TYPE, M, N, D0, n) creates the transfer function of
%   a lowpass filter, H, of the specified TYPE and size (M-by-N). To
%   view the filter as an image or mesh plot, it should be centered
%   using H = fftshift(H). 
%
%   Valid values for TYPE, D0, and n are:
%
%   'ideal'    Ideal lowpass filter with cutoff frequency D0. n need
%              not be supplied.  D0 must be positive.
%
%   'btw'      Butterworth lowpass filter of order n, and cutoff
%              D0.  The default value for n is 1.0.  D0 must be
%              positive.
%
%   'gaussian' Gaussian lowpass filter with cutoff (standard
%              deviation) D0.  n need not be supplied.  D0 must be
%              positive. 

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.8 $  $Date: 2004/11/04 22:33:16 $

% Use function dftuv to set up the meshgrid arrays needed for
% computing the required distances. 
[U, V] = dftuv(M, N);

% Compute the distances D(U, V).
D = sqrt(U.^2 + V.^2);

% Begin filter computations.
switch type
case 'ideal'
   H = double(D <= D0);
case 'btw'
   if nargin == 4
      n = 1;	
   end
   H = 1./(1 + (D./D0).^(2*n));
case 'gaussian'
   H = exp(-(D.^2)./(2*(D0^2)));
otherwise
   error('Unknown filter type.')
end
