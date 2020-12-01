function y = quantize(x, b, type)
%QUANTIZE Quantizes the elements of a UINT8 matrix.
%   Y = QUANTIZE(X, B, TYPE) quantizes X to B bits. Truncation is
%   used unless TYPE is 'igs' for Improved Gray Scale quantization. 

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.4 $  $Date: 2003/10/26 18:49:23 $

error(nargchk(2, 3, nargin));         % Check input arguments
if ndims(x) ~= 2 | ~isreal(x) | ...
     ~isnumeric(x) | ~isa(x, 'uint8')
   error('The input must be a UINT8 numeric matrix.');  
end

% Create bit masks for the quantization
lo = uint8(2 ^ (8 - b) - 1);
hi = uint8(2 ^ 8 - double(lo) - 1);

% Perform standard quantization unless IGS is specified
if nargin < 3 | ~strcmpi(type, 'igs')
   y = bitand(x, hi);
   
% Else IGS quantization. Process column-wise. If the MSB's of the
% pixel are all 1's, the sum is set to the pixel value. Else, add
% the pixel value to the LSB's of the previous sum. Then take the
% MSB's of the sum as the quantized value.
else
   [m, n] = size(x);                           s = zeros(m, 1);
   hitest = double(bitand(x, hi) ~= hi);       x = double(x);
   for j = 1:n
      s = x(:, j) + hitest(:, j) .* double(bitand(uint8(s), lo));
      y(:, j) = bitand(uint8(s), hi);
   end
end
