function rmse = compare(f1, f2, scale)
%COMPARE Computes and displays the error between two matrices.
%   RMSE = COMPARE(F1, F2, SCALE) returns the root-mean-square error
%   between inputs F1 and F2, displays a histogram of the difference,
%   and displays a scaled difference image. When SCALE is omitted, a
%   scale factor of 1 is used. 

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004.
%   $Revision: 1.3 $  $Date: 2003/04/18 05:07:33 $

% Check input arguments and set defaults.
error(nargchk(2, 3, nargin));
if nargin < 3     
   scale = 1;      
end

% Compute the root-mean-square error.
e = double(f1) - double(f2);
[m, n] = size(e);
rmse = sqrt(sum(e(:) .^ 2) / (m * n));

% Output error image & histogram if an error (i.e., rmse ~= 0).
if rmse
   % Form error histogram.
   emax = max(abs(e(:)));
   [h, x] = hist(e(:), emax);
   if length(h) >= 1
      figure;  bar(x, h, 'k');
      
      % Scale the error image symmetrically and display
      emax = emax / scale;
      e = mat2gray(e, [-emax, emax]);
      figure;   imshow(e);
   end
end
