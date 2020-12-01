function [r, R, S] = imnoise3(M, N, C, A, B)
%IMNOISE3 Generates periodic noise.
%    [r, R, S] = IMNOISE3(M, N, C, A, B), generates a spatial
%    sinusoidal noise pattern, r, of size M-by-N, its Fourier
%    transform, R, and spectrum, S.  The remaining parameters are:
%
%    C is a K-by-2 matrix with K pairs of freq. domain coordinates (u,
%    v) that define the locations of impulses in the freq. domain. The
%    locations are with respect to the frequency rectangle center at
%    (floor(M/2) + 1, floor(N/2) + 1).  The impulse locations are spe-
%    cified as increments with respect to the center. For ex, if M =
%    N = 512, then the center is at (257, 257). To specify an impulse
%    at (280, 300) we specify the pair (23, 43); i.e., 257 + 23 = 280,
%    and 257 + 43 = 300. Only one pair of coordinates is required for
%    each impulse.  The conjugate pairs are generated automatically. 
%
%    A is a 1-by-K vector that contains the amplitude of each of the
%    K impulse pairs. If A is not included in the argument, the
%    default used is A = ONES(1, K).  B is then automatically set to
%    its default values (see next paragraph).  The value specified
%    for A(j) is associated with the coordinates in C(j, 1:2). 
%
%    B is a K-by-2 matrix containing the Bx and By phase components
%    for each impulse pair.  The default values for B are B(1:K, 1:2)
%    = 0.
  
%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2004/11/04 22:32:42 $

% Process input parameters.
[K, n] = size(C);
if nargin == 3
   A(1:K) = 1.0;
   B(1:K, 1:2) = 0;
elseif nargin == 4
   B(1:K, 1:2) = 0;
end

% Generate R.
R = zeros(M, N);
for j = 1:K
   u1 = M/2 + 1 + C(j, 1); v1 = N/2 + 1 + C(j, 2);
   R(u1, v1) = i * (A(j)/2) * exp(i*2*pi*C(j, 1) * B(j, 1)/M);
   % Complex conjugate.
   u2 = M/2 + 1 - C(j, 1); v2 = N/2 + 1 - C(j, 2);
   R(u2, v2) = -i * (A(j)/2) * exp(i*2*pi*C(j, 2) * B(j, 2)/N);
end

% Compute spectrum and spatial sinusoidal pattern.
S = abs(R);
r = real(ifft2(ifftshift(R)));
