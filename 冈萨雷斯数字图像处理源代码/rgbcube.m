function rgbcube(vx, vy, vz)
%RGBCUBE Displays an RGB cube on the MATLAB desktop.
%   RGBCUBE(VX, VY, VZ) displays an RGB color cube, viewed from point
%   (VX, VY, VZ).  With no input arguments, RGBCUBE uses (10, 10, 4)
%   as the default viewing coordinates.  To view individual color
%   planes, use the following viewing coordinates, where the first
%   color in the sequence is the closest to the viewing axis, and the 
%   other colors are as seen from that axis, proceeding to the right
%   right (or above), and then moving clockwise. 
%
%      -------------------------------------------------
%           COLOR PLANE                  ( vx,  vy,  vz)
%      -------------------------------------------------
%       Blue-Magenta-White-Cyan          (  0,   0,  10)
%       Red-Yellow-White-Magenta         ( 10,   0,   0)
%       Green-Cyan-White-Yellow          (  0,  10,   0)
%       Black-Red-Magenta-Blue           (  0, -10,   0)
%       Black-Blue-Cyan-Green            (-10,   0,   0)
%       Black-Red-Yellow-Green           (  0,   0, -10)
%

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2003/10/13 00:52:14 $

% Set up parameters for function patch.
vertices_matrix = [0 0 0;0 0 1;0 1 0;0 1 1;1 0 0;1 0 1;1 1 0;1 1 1];
faces_matrix = [1 5 6 2;1 3 7 5;1 2 4 3;2 4 8 6;3 7 8 4;5 6 8 7];
colors = vertices_matrix; 
% The order of the cube vertices was selected to be the same as 
% the  order of the (R,G,B) colors (e.g., (0,0,0) corresponds to 
% black, (1,1,1) corresponds to white, and so on.)

% Generate RGB cube using function patch.
patch('Vertices', vertices_matrix, 'Faces', faces_matrix, ...
      'FaceVertexCData', colors, 'FaceColor', 'interp', ...
      'EdgeAlpha', 0) 

% Set up viewing point.
if nargin == 0
   vx = 10; vy = 10; vz = 4;
elseif nargin ~= 3
   error('Wrong number of inputs.')
end
axis off
view([vx, vy, vz])
axis square
