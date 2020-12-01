function lines = houghlines(f,theta,rho,rr,cc,fillgap,minlength)
%HOUGHLINES Extract line segments based on the Hough transform.
%   LINES = HOUGHLINES(F, THETA, RHO, RR, CC, FILLGAP, MINLENGTH)
%   extracts line segments in the image F associated with particular
%   bins in a Hough transform.  THETA and RHO are vectors returned by
%   function HOUGH.  Vectors RR and CC specify the rows and columns
%   of the Hough transform bins to use in searching for line
%   segments.  If HOUGHLINES finds two line segments associated with
%   the same Hough transform bin that are separated by less than
%   FILLGAP pixels, HOUGHLINES merges them into a single line
%   segment.  FILLGAP defaults to 20 if omitted.  Merged line
%   segments less than MINLENGTH pixels long are discarded.
%   MINLENGTH defaults to 40 if omitted. 
%
%   LINES is a structure array whose length equals the number of
%   merged line segments found.  Each element of the structure array
%   has these fields: 
%
%      point1    End-point of the line segment; two-element vector
%      point2    End-point of the line segment; two-element vector
%      length    Distance between point1 and point2
%      theta     Angle (in degrees) of the Hough transform bin
%      rho       Rho-axis position of the Hough transform bin

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.4 $  $Date: 2003/10/26 22:34:10 $

if nargin < 6 
   fillgap = 20; 
end
if nargin < 7 
   minlength = 40; 
end

numlines = 0; lines = struct;
for k = 1:length(rr)
   rbin = rr(k); cbin = cc(k);
   
   % Get all pixels associated with Hough transform cell.
   [r, c] = houghpixels(f, theta, rho, rbin, cbin);
   if isempty(r) 
      continue 
   end
   
   % Rotate the pixel locations about (1,1) so that they lie
   % approximately along a vertical line.
   omega = (90 - theta(cbin)) * pi / 180;
   T = [cos(omega) sin(omega); -sin(omega) cos(omega)];
   xy = [r - 1  c - 1] * T;
   x = sort(xy(:,1));
   
   % Find the gaps larger than the threshold.
   diff_x = [diff(x); Inf];
   idx = [0; find(diff_x > fillgap)];
   for p = 1:length(idx) - 1
      x1 = x(idx(p) + 1); x2 = x(idx(p + 1));
      linelength = x2 - x1;
      if linelength >= minlength
         point1 = [x1 rho(rbin)]; point2 = [x2 rho(rbin)];
         % Rotate the end-point locations back to the original
         % angle.
         Tinv = inv(T);
         point1 = point1 * Tinv; point2 = point2 * Tinv;
         
         numlines = numlines + 1;
         lines(numlines).point1 = point1 + 1;
         lines(numlines).point2 = point2 + 1;
         lines(numlines).length = linelength;
         lines(numlines).theta = theta(cbin);
         lines(numlines).rho = rho(rbin);
      end
   end
end
