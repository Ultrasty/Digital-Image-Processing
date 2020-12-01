function [h, theta, rho] = hough(f, dtheta, drho)
%HOUGH Hough transform.
%   [H, THETA, RHO] = HOUGH(F, DTHETA, DRHO) computes the Hough
%   transform of the image F.  DTHETA specifies the spacing (in
%   degrees) of the Hough transform bins along the theta axis.  DRHO
%   specifies the spacing of the Hough transform bins along the rho
%   axis.  H is the Hough transform matrix.  It is NRHO-by-NTHETA,
%   where NRHO = 2*ceil(norm(size(F))/DRHO) - 1, and NTHETA =
%   2*ceil(90/DTHETA).  Note that if 90/DTHETA is not an integer, the
%   actual angle spacing will be 90 / ceil(90/DTHETA).
%
%   THETA is an NTHETA-element vector containing the angle (in
%   degrees) corresponding to each column of H.  RHO is an
%   NRHO-element vector containing the value of rho corresponding to
%   each row of H. 
%
%   [H, THETA, RHO] = HOUGH(F) computes the Hough transform using
%   DTHETA = 1 and DRHO = 1. 

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.4 $  $Date: 2003/10/26 22:33:44 $

if nargin < 3
   drho = 1;
end
if nargin < 2
   dtheta = 1;
end

f = double(f);
[M,N] = size(f);
theta = linspace(-90, 0, ceil(90/dtheta) + 1);
theta = [theta -fliplr(theta(2:end - 1))];
ntheta = length(theta);

D = sqrt((M - 1)^2 + (N - 1)^2);
q = ceil(D/drho);
nrho = 2*q - 1;
rho = linspace(-q*drho, q*drho, nrho);

[x, y, val] = find(f);
x = x - 1;  y = y - 1;

% Initialize output.
h = zeros(nrho, length(theta));

% To avoid excessive memory usage, process 1000 nonzero pixel
% values at a time.
for k = 1:ceil(length(val)/1000)
   first = (k - 1)*1000 + 1;
   last  = min(first+999, length(x));
   
   x_matrix     = repmat(x(first:last), 1, ntheta);
   y_matrix     = repmat(y(first:last), 1, ntheta);
   val_matrix   = repmat(val(first:last), 1, ntheta);
   theta_matrix = repmat(theta, size(x_matrix, 1), 1)*pi/180;
   
   rho_matrix = x_matrix.*cos(theta_matrix) + ...
       y_matrix.*sin(theta_matrix);
   slope = (nrho - 1)/(rho(end) - rho(1));
   rho_bin_index = round(slope*(rho_matrix - rho(1)) + 1);
   
   theta_bin_index = repmat(1:ntheta, size(x_matrix, 1), 1);
   
   % Take advantage of the fact that the SPARSE function, which
   % constructs a sparse matrix, accumulates values when input
   % indices are repeated.  That’s the behavior we want for the
   % Hough transform.  We want the output to be a full (nonsparse)
   % matrix, however, so we call function FULL on the output of
   % SPARSE.
   h = h + full(sparse(rho_bin_index(:), theta_bin_index(:), ...
                       val_matrix(:), nrho, ntheta));
end
