function [varargout] = waveback(c, s, varargin)
%WAVEBACK Performs a multi-level two-dimensional inverse FWT.
%   [VARARGOUT] = WAVEBACK(C, S, VARARGIN) computes a 2D N-level
%   partial or complete wavelet reconstruction of decomposition
%   structure [C, S]. 
%
%   SYNTAX:
%   Y = WAVEBACK(C, S, 'WNAME');  Output inverse FWT matrix Y 
%   Y = WAVEBACK(C, S, LR, HR);   using lowpass and highpass 
%                                 reconstruction filters (LR and 
%                                 HR) or filters obtained by 
%                                 calling WAVEFILTER with 'WNAME'.
%
%   [NC, NS] = WAVEBACK(C, S, 'WNAME', N);  Output new wavelet 
%   [NC, NS] = WAVEBACK(C, S, LR, HR, N);   decomposition structure
%                                           [NC, NS] after N step 
%                                           reconstruction.
%
%   See also WAVEFAST and WAVEFILTER.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2003/10/13 01:29:36 $

% Check the input and output arguments for reasonableness.
error(nargchk(3, 5, nargin));
error(nargchk(1, 2, nargout));
   
if (ndims(c) ~= 2) | (size(c, 1) ~= 1)
  error('C must be a row vector.');   
end
  
if (ndims(s) ~= 2) | ~isreal(s) | ~isnumeric(s) | (size(s,2) ~= 2)
  error('S must be a real, numeric two-column array.');   
end
  
elements = prod(s, 2);
if (length(c) < elements(end)) | ...
      ~(elements(1) + 3 * sum(elements(2:end - 1)) >= elements(end))
  error(['[C S] must be a standard wavelet ' ...
         'decomposition structure.']); 
end

% Maximum levels in [C, S].
nmax = size(s, 1) - 2;        
% Get third input parameter and init check flags.
wname = varargin{1};  filterchk = 0;   nchk = 0;    
  
switch nargin
case 3
   if ischar(wname)   
      [lp, hp] = wavefilter(wname, 'r');   n = nmax;
   else 
      error('Undefined filter.');  
   end
   if nargout ~= 1 
      error('Wrong number of output arguments.');  
   end
case 4
   if ischar(wname)
      [lp, hp] = wavefilter(wname, 'r');   
      n = varargin{2};    nchk = 1;
   else
      lp = varargin{1};   hp = varargin{2};   
      filterchk = 1;   n = nmax;
      if nargout ~= 1 
         error('Wrong number of output arguments.');  
      end
   end
case 5
    lp = varargin{1};   hp = varargin{2};   filterchk = 1;
    n = varargin{3};    nchk = 1;
otherwise
    error('Improper number of input arguments.');     
end
  
fl = length(lp);
if filterchk                                        % Check filters.
  if (ndims(lp) ~= 2) | ~isreal(lp) | ~isnumeric(lp) ...
        | (ndims(hp) ~= 2) | ~isreal(hp) | ~isnumeric(hp) ...
        | (fl ~= length(hp)) | rem(fl, 2) ~= 0
      error(['LP and HP must be even and equal length real, ' ...
             'numeric filter vectors.']); 
  end
end

if nchk & (~isnumeric(n) | ~isreal(n))          % Check scale N.
    error('N must be a real numeric.'); 
end
if (n > nmax) | (n < 1)
   error('Invalid number (N) of reconstructions requested.');    
end
if (n ~= nmax) & (nargout ~= 2) 
   error('Not enough output arguments.'); 
end
  
nc = c;    ns = s;    nnmax = nmax;             % Init decomposition.
for i = 1:n
   % Compute a new approximation.
   a = symconvup(wavecopy('a', nc, ns), lp, lp, fl, ns(3, :)) + ...
       symconvup(wavecopy('h', nc, ns, nnmax), ...
                 hp, lp, fl, ns(3, :)) + ...
       symconvup(wavecopy('v', nc, ns, nnmax), ...
                 lp, hp, fl, ns(3, :)) + ...
       symconvup(wavecopy('d', nc, ns, nnmax), ...
                 hp, hp, fl, ns(3, :));
      
    % Update decomposition.
    nc = nc(4 * prod(ns(1, :)) + 1:end);     nc = [a(:)' nc];
    ns = ns(3:end, :);                       ns = [ns(1, :); ns];
    nnmax = size(ns, 1) - 2;
end

% For complete reconstructions, reformat output as 2-D.
if nargout == 1
    a = nc;   nc = repmat(0, ns(1, :));     nc(:) = a;    
end
  
varargout{1} = nc;
if nargout == 2 
   varargout{2} = ns;  
end

%-------------------------------------------------------------------%
function z = symconvup(x, f1, f2, fln, keep)
% Upsample rows and convolve columns with f1; upsample columns and
% convolve rows with f2; then extract center assuming symmetrical
% extension.

y = zeros([2 1] .* size(x));      y(1:2:end, :) = x;
y = conv2(y, f1');
z = zeros([1 2] .* size(y));      z(:, 1:2:end) = y;
z = conv2(z, f2);
z = z(fln - 1:fln + keep(1) - 2, fln - 1:fln + keep(2) - 2);
