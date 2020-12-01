function x = jpeg2k2im(y)
%JPEG2K2IM Decodes an IM2JPEG2K compressed image.
%   X = JPEG2K2IM(Y) decodes compressed image Y, reconstructing an
%   approximation of the original image X.  Y is an encoding
%   structure returned by IM2JPEG2K. 
%
%   See also IM2JPEG2K.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.4 $  $Date: 2003/10/26 18:39:40 $

error(nargchk(1, 1, nargin));       % Check input arguments

% Get decoding parameters: scale, quantization vector, run-length
% table size, zero run code, end-of-data code, wavelet bookkeeping
% array, and run-length table.
n = double(y.n);
q = double(y.q) / 100;
runs = double(y.runs);
rlen = length(runs);
zrc = -double(y.zrc);
eoc = zrc - 1;
s = double(y.s);
s = reshape(s, n + 2, 2);

% Compute the size of the wavelet transform.
cl = prod(s(1, :));
for i = 2:n + 1
   cl = cl + 3 * prod(s(i, :));     
end

% Perform Huffman decoding followed by zero run decoding.
r = huff2mat(y.huffman);

c = [];   zi = find(r == zrc);    i = 1;
for j = 1:length(zi)
   c = [c r(i:zi(j) - 1) zeros(1, runs(r(zi(j) + 1)))];
   i = zi(j) + 2;      
end

zi = find(r == eoc);                % Undo terminating zero run
if length(zi) == 1                  % or last non-zero run.
   c = [c r(i:zi - 1)];
   c = [c zeros(1, cl - length(c))];
else
   c = [c r(i:end)];  
end

% Denormalize the coefficients.
c = c + (c > 0) - (c < 0);
for k = 1:n
   qi = 3 * k - 2;
   c = wavepaste('h', c, s, k, wavecopy('h', c, s, k) * q(qi));
   c = wavepaste('v', c, s, k, wavecopy('v', c, s, k) * q(qi + 1));
   c = wavepaste('d', c, s, k, wavecopy('d', c, s, k) * q(qi + 2));
end
c = wavepaste('a', c, s, k, wavecopy('a', c, s, k) * q(qi + 3));

% Compute the inverse wavelet transform and level shift.
x = waveback(c, s, 'jpeg9.7', n);
x = uint8(x + 128);