function [B, theta] = x2majoraxis(A, B, type)
%X2MAJORAXIS Aligns coordinate x with the major axis of a region.
%   [B2, THETA] = X2MAJORAXIS(A, B, TYPE) aligns the x-coordinate
%   axis with the major axis of a region or boundary. The y-axis is
%   perpendicular to the x-axis.  The rows of 2-by-2 matrix A are the
%   coordinates of the two end points of the major axis, in the form
%   A = [x1 y1; x2 y2]. On input, B is either a binary image (i.e.,
%   an array of class logical) containing a single region, or it is
%   an np-by-2 set of points representing a (connected) boundary. In
%   the latter case, the first column of B must represent
%   x-coordinates and the second column must represent the
%   corresponding y-coordinates. On output, B2 contains the same data
%   as the input, but aligned with the major axis. If the input is an
%   image, so is the output; similarly the output is a sequence of
%   coordinates if the input is such a sequence. Parameter THETA is
%   the initial angle between the major axis and the x-axis. The
%   origin of the xy-axis system is at the bottom left; the x-axis is
%   the horizontal axis and the y-axis is the vertical. 
%
%   Keep in mind that rotations can introduce round-off errors when
%   the data are converted to integer coordinates, which is a
%   requirement.  Thus, postprocessing (e.g., with bwmorph) of the
%   output may be required to reconnect a boundary.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.9 $  $Date: 2003/11/21 14:51:54 $

% Preliminaries.
if islogical(B)
   type = 'region';
elseif size(B, 2) == 2
   type = 'boundary';
   [M, N] = size(B);
   if M < N
      error('B is boundary. It must be of size np-by-2; np > 2.')
   end
   % Compute centroid for later use. c is a 1-by-2 vector. 
   % Its 1st component is the mean of the boundary in the x-direction.
   % The second is the mean in the y-direction.
   c(1) = round((min(B(:, 1)) + max(B(:, 1))/2));
   c(2) = round((min(B(:, 2)) + max(B(:, 2))/2));
   
   % It is possible for a connected boundary to develop small breaks
   % after rotation. To prevent this, the input boundary is filled, 
   % processed as a region, and then the boundary is re-extracted. This
   % guarantees that the output will be a connected boundary.
   m = max(size(B));
   % The following image is of size m-by-m to make sure that there
   % there will be no size truncation after rotation.
   B = bound2im(B,m,m); 
   B = imfill(B,'holes');
else
   error('Input must be a boundary or a binary image.')
end

% Major axis in vector form.
v(1) = A(2, 1) - A(1, 1);
v(2) = A(2, 2) - A(1, 2);
v = v(:);  % v is a col vector

% Unit vector along x-axis.
u = [1; 0];

% Find angle between major axis and x-axis. The angle is
% given by acos of the inner product of u and v divided by
% the product of their norms. Because the inputs are image
% points, they are in the first quadrant.
nv = norm(v);
nu = norm(u);
theta = acos(u'*v/nv*nu); 
if theta > pi/2
   theta = -(theta - pi/2);
end
theta = theta*180/pi;  % Convert angle to degrees.

% Rotate by angle theta and crop the rotated image to original size.
B = imrotate(B, theta, 'bilinear', 'crop');

% If the input was a boundary, re-extract it.
if  strcmp(type, 'boundary')
   B = boundaries(B);
   B = B{1};
   % Shift so that centroid of the extracted boundary is  
   % approx equal to the centroid of the original boundary:
   B(:, 1) = B(:, 1) - min(B(:, 1)) + c(1);
   B(:, 2) = B(:, 2) - min(B(:, 2)) + c(2);
end
