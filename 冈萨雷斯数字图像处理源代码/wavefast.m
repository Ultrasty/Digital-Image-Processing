function [c, s] = wavefast(x, n, varargin)
%WAVEFAST Perform multi-level 2-dimensional fast wavelet transform.
%   [C, L] = WAVEFAST(X, N, LP, HP) performs a 2D N-level FWT of
%   image (or matrix) X with respect to decomposition filters LP and
%   HP.
%
%   [C, L] = WAVEFAST(X, N, WNAME) performs the same operation but
%   fetches filters LP and HP for wavelet WNAME using WAVEFILTER.
%
%   Scale parameter N must be less than or equal to log2 of the
%   maximum image dimension.  Filters LP and HP must be even. To
%   reduce border distortion, X is symmetrically extended. That is,
%   if X = [c1 c2 c3 ... cn] (in 1D), then its symmetric extension
%   would be [... c3 c2 c1 c1 c2 c3 ... cn cn cn-1 cn-2 ...].
%
%   OUTPUTS:
%     Matrix C is a coefficient decomposition vector:
%
%      C = [ a(n) h(n) v(n) d(n) h(n-1) ... v(1) d(1) ]
%
%     where a, h, v, and d are columnwise vectors containing
%     approximation, horizontal, vertical, and diagonal coefficient
%     matrices, respectively.  C has 3n + 1 sections where n is the
%     number of wavelet decompositions. 
%
%     Matrix S is an (n+2) x 2 bookkeeping matrix:
%
%      S = [ sa(n, :); sd(n, :); sd(n-1, :); ... ; sd(1, :); sx ]
%
%     where sa and sd are approximation and detail size entries.
%
%   See also WAVEBACK and WAVEFILTER.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2003/10/13 01:14:17 $

% Check the input arguments for reasonableness.
error(nargchk(3, 4, nargin));

if nargin == 3
   if ischar(varargin{1})   
      [lp, hp] = wavefilter(varargin{1}, 'd');
   else 
      error('Missing wavelet name.');   
   end
else
      lp = varargin{1};     hp = varargin{2};   
end

fl = length(lp);      sx = size(x);

if (ndims(x) ~= 2) | (min(sx) < 2) | ~isreal(x) | ~isnumeric(x)
   error('X must be a real, numeric matrix.');     
end
  
if (ndims(lp) ~= 2) | ~isreal(lp) | ~isnumeric(lp) ...
       | (ndims(hp) ~= 2) | ~isreal(hp) | ~isnumeric(hp) ...
       | (fl ~= length(hp)) | rem(fl, 2) ~= 0
   error(['LP and HP must be even and equal length real, ' ...
          'numeric filter vectors.']); 
end
  
if ~isreal(n) | ~isnumeric(n) | (n < 1) | (n > log2(max(sx)))
   error(['N must be a real scalar between 1 and ' ...
          'log2(max(size((X))).']);    
end
  
% Init the starting output data structures and initial approximation.
c = [];       s = sx;     app = double(x);

% For each decomposition ...
for i = 1:n
   % Extend the approximation symmetrically.
   [app, keep] = symextend(app, fl);
   
   % Convolve rows with HP and downsample. Then convolve columns
   % with HP and LP to get the diagonal and vertical coefficients.
   rows = symconv(app, hp, 'row', fl, keep);
   coefs = symconv(rows, hp, 'col', fl, keep);
   c = [coefs(:)' c];    s = [size(coefs); s];
   coefs = symconv(rows, lp, 'col', fl, keep);
   c = [coefs(:)' c];
   
   % Convolve rows with LP and downsample. Then convolve columns
   % with HP and LP to get the horizontal and next approximation
   % coeffcients.
   rows = symconv(app, lp, 'row', fl, keep);
   coefs = symconv(rows, hp, 'col', fl, keep);
   c = [coefs(:)' c];
   app = symconv(rows, lp, 'col', fl, keep);
end

% Append final approximation structures.
c = [app(:)' c];       s = [size(app); s];

%-------------------------------------------------------------------%
function [y, keep] = symextend(x, fl)
% Compute the number of coefficients to keep after convolution
% and downsampling. Then extend x in both dimensions.

keep = floor((fl + size(x) - 1) / 2);
y = padarray(x, [(fl - 1) (fl - 1)], 'symmetric', 'both');

%-------------------------------------------------------------------%
function y = symconv(x, h, type, fl, keep)
% Convolve the rows or columns of x with h, downsample,
% and extract the center section since symmetrically extended.

if strcmp(type, 'row')
   y = conv2(x, h);
   y = y(:, 1:2:end);
   y = y(:, fl / 2 + 1:fl / 2 + keep(2));
else
   y = conv2(x, h');
   y = y(1:2:end, :);
   y = y(fl / 2 + 1:fl / 2 + keep(1), :);
end
