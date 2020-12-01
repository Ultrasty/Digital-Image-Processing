function s = subim(f, m, n, rx, cy)
%SUBIM Extract subimage.
%   S = SUBIM(F, M, N, RX, CY) extracts a subimage, S, from the input
%   image, F.  The subimage is of size M-by-N, and the coordinates of
%   its top, left corner are (RX, CY).
%
%   Sample M-file used in Chapter 2.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.1 $  $Date: 2003/02/22 17:00:54 $

s = zeros(m, n);
rowhigh = rx + m - 1;
colhigh = cy + n - 1;
xcount = 0;
for r = rx:rowhigh
   xcount = xcount + 1;
   ycount = 0;
   for c = cy:colhigh
      ycount = ycount + 1;
      s(xcount, ycount) = f(r, c);
   end
end
