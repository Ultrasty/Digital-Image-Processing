function [p, pmax, pmin, pn] = improd(f, g)
%IMPROD Compute the product of two images.
%   [P, PMAX, PMIN, PN] = IMPROD(F, G) outputs the element-by-element
%   product of two input images, F and G, the product maximum and
%   minimum values, and a normalized product array with values in the
%   range [0, 1]. The input images must be of the same size.  They
%   can be of class uint8, unit16, or double. The outputs are of
%   class double.
%
%   Sample M-file used in Chapter 2.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.1 $  $Date: 2003/02/22 17:00:54 $

fd = double(f);
gd = double(g);
p = fd.*gd;
pmax = max(p(:));
pmin = min(p(:));
pn = mat2gray(p); 					
