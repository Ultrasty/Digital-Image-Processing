function [s, su] = bsubsamp(b, gridsep)
%BSUBSAMP Subsample a boundary.
%   [S, SU] = BSUBSAMP(B, GRIDSEP) subsamples the boundary B by
%   assigning each of its points to the grid node to which it is
%   closest.  The grid is specified by GRIDSEP, which is the
%   separation in pixels between the grid lines. For example, if
%   GRIDSEP = 2, there are two pixels in between grid lines. So, for
%   instance, the grid points in the first row would be at (1,1),
%   (1,4), (1,6), ..., and similarly in the y direction. The value
%   of GRIDSEP must be an even integer. The boundary is specified by
%   a set of coordinates in the form of an np-by-2 array.  It is
%   assumed that the boundary is one pixel thick. 
%
%   Output S is the subsampled boundary. Output SU is normalized so
%   that the grid separation is unity.  This is useful for obtaining
%   the Freeman chain code of the subsampled boundary.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.8 $  $Date: 2004/11/04 20:17:59 $
 
% Check input.
[np, nc] = size(b);
if np < nc 
   error('B must be of size np-by-2.'); 
end
if gridsep/2 ~= round(gridsep/2) 
   error('GRIDSEP must be an even integer.')
end

% Some boundary tracing programs, such as boundaries.m, end with 
% the beginning, resulting in a sequence in which the coordinates
% of the first and last points are the same. If this is the case 
% in b, eliminate the last point.
if isequal(b(1, :), b(np, :))
   np = np - 1;
   b = b(1:np, :);
end

% Find the max x and y spanned by the boundary.
xmax = max(b(:, 1));
ymax = max(b(:, 2));

% Determine the integral number of grid lines with gridsep points in
% between them that encompass the intervals [1,xmax], [1,ymax].
GLx = ceil((xmax + gridsep)/(gridsep + 1));
GLy = ceil((ymax + gridsep)/(gridsep + 1));

% Form vectors of x and y grid locations.
I = 1:GLx;
% Vector of grid line locations intersecting x-axis.
X(I) = gridsep*I + (I - gridsep); 

J = 1:GLy;
% Vector of grid line locations intersecting y-axis.
Y(J) = gridsep*J + (J - gridsep); 

% Compute both components of the cityblock distance between each
% element of b and all the grid-line intersections.  Assign each
% point to the grid location for which each comp of the cityblock
% distance was <= gridsep/2. Note the use of meshgrid to
% optimize the code. Keep in mind that meshgrid requires that columns
% be listed first (see Chapter 2).
DIST = gridsep/2;
[YG, XG] = meshgrid(Y, X);
Q = 1;
for k=1:np
   [I,J] = find(abs(XG - b(k, 1)) <= DIST & abs(YG - b(k, 2)) <= ...
                DIST); 
   % If point b(k,:) is equidistant from two or more grid intersections,
   % assign the point arbitrarily to the first one:
   I = I(1);
   J = J(1);
   ord = k; % To keep track of order of input coordinates.
   d1(Q, :) = cat(2, Y(J), ord);
   d2(Q, :) = cat(2, X(I), ord);
   Q = Q + 1;
end
 
% d is the set of points assigned to the new grid with line
% separation of gridsep. Note that it is formed as d=(d2,d1) to
% compensate for the coordinate transposition inherent in using
% meshgrid (see Chapter 2). 
d = cat(2, d2(:, 1), d1); % The second column of d1 & d2 is ord.

% Sort the points using the values in ord, which is the last col in
% d.
d = fliplr(d); % So the last column becomes first.
d = sortrows(d);
d = fliplr(d); % Flip back.

% Eliminate duplicate rows in the first two components of 
% d to create the output. The cw or ccw order MUST be preserved.
s = d(:, 1:2);
[s, m, n] = unique(s, 'rows');

% Function unique sorts the data--Restore to original order
% by using the contents of m.
s = [s, m];
s = fliplr(s);
s = sortrows(s);
s = fliplr(s);
s = s(:, 1:2);

% Scale to unit grid so that can use directly to obtain Freeman
% chain code.  The shape does not change.
su = round(s./gridsep) + 1;
