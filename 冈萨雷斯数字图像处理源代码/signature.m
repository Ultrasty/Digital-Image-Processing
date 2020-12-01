function [st, angle, x0, y0] = signature(b, varargin)
%SIGNATURE Computes the signature of a boundary. 
%   [ST, ANGLE, X0, Y0] = SIGNATURE(B) computes the signature of a
%   given boundary, B, where B is an np-by-2 array (np > 2)
%   containing the (x, y) coordinates of the boundary ordered in a
%   clockwise or counterclockwise direction. The amplitude of the
%   signature as a function of increasing ANGLE is output in
%   ST. (X0,Y0) are the coordinates of the centroid of the
%   boundary. The maximum size of arrays ST and ANGLE is 360-by-1,
%   indicating a maximum resolution of one degree. The input must be
%   a one-pixel-thick boundary obtained, for example, by using the
%   function boundaries. By definition, a boundary is a closed
%   curve. 
% 
%   [ST, ANGLE, X0, Y0] = SIGNATURE(B) computes the signature, using
%   the centroid as the origin of the signature vector. 
%   
%   [ST, ANGLE, X0, Y0] = SIGNATURE(B, X0, Y0) computes the boundary
%   using the specified (X0, Y0) as the origin of the signature
%   vector.   

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.6 $  $Date: 2003/11/21 14:46:47 $

% Check dimensions of b.
[np, nc] = size(b);
if (np < nc | nc ~= 2) 
   error('B must be of size np-by-2.'); 
end

% Some boundary tracing programs, such as boundaries.m, end where 
% they started, resulting in a sequence in which the coordinates
% of the first and last points are the same. If this is the case, 
% in b, eliminate the last point.
if isequal(b(1, :), b(np, :)) 
   b = b(1:np - 1, :); 
   np = np - 1;
end
    
% Compute parameters.
if nargin == 1
   x0 = round(sum(b(:, 1))/np); % Coordinates of the centroid.
   y0 = round(sum(b(:, 2))/np); 
elseif nargin == 3 
   x0 = varargin{1};
   y0 = varargin{2};
else 
   error('Incorrect number of inputs.'); 
end
    
% Shift origin of coord system to (x0, y0)).
b(:, 1) = b(:, 1) - x0;
b(:, 2) = b(:, 2) - y0;

% Convert the coordinates to polar.  But first have to convert the
% given image coordinates, (x, y), to the coordinate system used by
% MATLAB for conversion between Cartesian and polar cordinates.
% Designate these coordinates by (xc, yc). The two coordinate systems
% are related as follows:  xc = y and yc = -x.
xc = b(:, 2);
yc = -b(:, 1);
[theta, rho] = cart2pol(xc, yc);

% Convert angles to degrees.
theta = theta.*(180/pi);

% Convert to all nonnegative angles.
j = theta == 0; % Store the indices of theta = 0 for use below.
theta = theta.*(0.5*abs(1 + sign(theta)))...
        - 0.5*(-1 + sign(theta)).*(360 + theta);
theta(j) = 0; % To preserve the 0 values.

temp = theta;
% Order temp so that sequence starts with the smallest angle.
% This will be used below in a check for monotonicity.
I = find(temp == min(temp));

% Scroll up so that sequence starts with the smallest angle.
% Use I(1) in case the min is not unique (in this case the
% sequence will not be monotonic anyway).
temp = circshift(temp, [-(I(1) - 1), 0]);

% Check for monotonicity, and issue a warning if sequence
% is not monotonic. First determine if sequence is 
% cw or ccw.
k1 = abs(temp(1) - temp(2));
k2 = abs(temp(1) - temp(3));
if k2 > k1
   sense = 1; % ccw
elseif k2 < k1
   sense = -1; % cw
else
   warning(['The first 3 points in B do not form a monotonic ' ...
            'sequence.']);
end
% Check the rest of the sequence for monotonicity. Because 
% the angles are rounded to the nearest integer later in the 
% program, only differences greater than 0.5 degrees are 
% considered in the test for monotonicity in the rest of
% the sequence.
flag = 0;
for k = 3:length(temp) - 1
   diff = sense*(temp(k + 1) - temp(k));
   if diff < -.5 
      flag = 1;
   end
end
if flag
   warning('Angles do not form a monotonic sequence.');
end

% Round theta to 1 degree increments.
theta = round(theta);

% Keep theta and rho together.
tr = [theta, rho]; 

% Delete duplicate angles.  The unique operation
% also sorts the input in ascending order.
[w, u, v] = unique(tr(:, 1)); 
tr = tr(u,:); % u identifies the rows kept by unique.

% If the last angle equals 360 degrees plus the first
% angle, delete the last angle.
if tr(end, 1) == tr(1) + 360
   tr = tr(1:end - 1, :);
end
        
% Output the angle values.
angle = tr(:, 1);

% The signature is the set of values of rho corresponding 
% to the angle values.
st = tr(:, 2); 
