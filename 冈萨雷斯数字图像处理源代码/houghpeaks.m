function [r, c, hnew] = houghpeaks(h, numpeaks, threshold, nhood)
%HOUGHPEAKS Detect peaks in Hough transform.
%   [R, C, HNEW] = HOUGHPEAKS(H, NUMPEAKS, THRESHOLD, NHOOD) detects
%   peaks in the Hough transform matrix H.  NUMPEAKS specifies the
%   maximum number of peak locations to look for.  Values of H below
%   THRESHOLD will not be considered to be peaks.  NHOOD is a
%   two-element vector specifying the size of the suppression
%   neighborhood.  This is the neighborhood around each peak that is
%   set to zero after the peak is identified.  The elements of NHOOD
%   must be positive, odd integers.  R and C are the row and column
%   coordinates of the identified peaks.  HNEW is the Hough transform
%   with peak neighborhood suppressed. 
%
%   If NHOOD is omitted, it defaults to the smallest odd values >=
%   size(H)/50.  If THRESHOLD is omitted, it defaults to
%   0.5*max(H(:)).  If NUMPEAKS is omitted, it defaults to 1. 

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2003/11/21 13:34:50 $

if nargin < 4
   nhood = size(h)/50;
   % Make sure the neighborhood size is odd.
   nhood = max(2*ceil(nhood/2) + 1, 1);
end
if nargin < 3
   threshold = 0.5 * max(h(:));
end
if nargin < 2
   numpeaks = 1;
end

done = false;
hnew = h; r = []; c = [];
while ~done
   [p, q] = find(hnew == max(hnew(:)));
   p = p(1); q = q(1);
   if hnew(p, q) >= threshold
      r(end + 1) = p; c(end + 1) = q;

      % Suppress this maximum and its close neighbors.
      p1 = p - (nhood(1) - 1)/2; p2 = p + (nhood(1) - 1)/2;
      q1 = q - (nhood(2) - 1)/2; q2 = q + (nhood(2) - 1)/2;
      [pp, qq] = ndgrid(p1:p2,q1:q2);
      pp = pp(:); qq = qq(:);

      % Throw away neighbor coordinates that are out of bounds in
      % the rho direction.
      badrho = find((pp < 1) | (pp > size(h, 1)));
      pp(badrho) = []; qq(badrho) = [];

      % For coordinates that are out of bounds in the theta
      % direction, we want to consider that H is antisymmetric
      % along the rho axis for theta = +/- 90 degrees.
      theta_too_low = find(qq < 1);
      qq(theta_too_low) = size(h, 2) + qq(theta_too_low);
      pp(theta_too_low) = size(h, 1) - pp(theta_too_low) + 1;
      theta_too_high = find(qq > size(h, 2));
      qq(theta_too_high) = qq(theta_too_high) - size(h, 2);
      pp(theta_too_high) = size(h, 1) - pp(theta_too_high) + 1;

      % Convert to linear indices to zero out all the values.
      hnew(sub2ind(size(hnew), pp, qq)) = 0;

      done = length(r) == numpeaks;
   else
      done = true;
   end
end
