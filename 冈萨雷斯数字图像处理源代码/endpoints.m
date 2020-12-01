function g = endpoints(f)
%ENDPOINTS Computes end points of a binary image.
%   G = ENDPOINTS(F) computes the end points of the binary image F
%   and returns them in the binary image G. 

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.3 $  $Date: 2003/04/22 01:18:18 $

persistent lut

if isempty(lut)
   lut = makelut(@endpoint_fcn, 3);
end

g = applylut(f,lut);

%-------------------------------------------------------------------%
function is_end_point = endpoint_fcn(nhood)
%   Determines if a pixel is an end point.
%   IS_END_POINT = ENDPOINT_FCN(NHOOD) accepts a 3-by-3 binary
%   neighborhood, NHOOD, and returns a 1 if the center element is an
%   end point; otherwise it returns a 0. 

is_end_point = nhood(2,2) & (sum(nhood(:)) == 2);
