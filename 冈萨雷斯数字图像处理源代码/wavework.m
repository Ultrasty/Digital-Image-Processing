function [varargout] = wavework(opcode, type, c, s, n, x)
%WAVEWORK is used to edit wavelet decomposition structures.
%   [VARARGOUT] = WAVEWORK(OPCODE, TYPE, C, S, N, X) gets the
%   coefficients specified by TYPE and N for access or modification
%   based on OPCODE.
%
%   INPUTS:
%     OPCODE      Operation to perform
%     --------------------------------------------------------------
%     'copy'      [varargout] = Y = requested (via TYPE and N)
%                 coefficient matrix 
%     'cut'       [varargout] = [NC, Y] = New decomposition vector
%                 (with requested coefficient matrix zeroed) AND 
%                 requested coefficient matrix 
%     'paste'     [varargout] = [NC] = new decomposition vector with
%                 coefficient matrix replaced by X
%
%     TYPE        Coefficient category
%     --------------------------------------------
%     'a'         Approximation coefficients
%     'h'         Horizontal details
%     'v'         Vertical details
%     'd'         Diagonal details
%
%     [C, S] is a wavelet toolbox decomposition structure.
%     N is a decomposition level (Ignored if TYPE = 'a').
%     X is a two-dimensional coefficient matrix for pasting.
%
%   See also WAVECUT, WAVECOPY, and WAVEPASTE.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.6 $  $Date: 2003/11/21 15:00:18 $

error(nargchk(4, 6, nargin));

if (ndims(c) ~= 2) | (size(c, 1) ~= 1)
   error('C must be a row vector.');   
end
  
if (ndims(s) ~= 2) | ~isreal(s) | ~isnumeric(s) | (size(s, 2) ~= 2)
   error('S must be a real, numeric two-column array.');   
end
  
elements = prod(s, 2);                % Coefficient matrix elements.
if (length(c) < elements(end)) | ...
      ~(elements(1) + 3 * sum(elements(2:end - 1)) >= elements(end))
    error(['[C S] must form a standard wavelet decomposition ' ...
           'structure.']); 
end

if strcmp(lower(opcode(1:3)), 'pas') & nargin < 6
   error('Not enough input arguments.');   
end
  
if nargin < 5
   n = 1;      % Default level is 1.
end             
nmax = size(s, 1) - 2;                % Maximum levels in [C, S].

aflag = (lower(type(1)) == 'a');
if ~aflag & (n > nmax)
   error('N exceeds the decompositions in [C, S].');    
end
  
switch lower(type(1))                 % Make pointers into C.
case 'a'
   nindex = 1;
   start = 1;    stop = elements(1);    ntst = nmax;
case  {'h', 'v', 'd'}
   switch type
   case 'h', offset = 0;     % Offset to details.
   case 'v', offset = 1;
   case 'd', offset = 2;
   end
   nindex = size(s, 1) - n;      % Index to detail info.
   start = elements(1) + 3 * sum(elements(2:nmax - n + 1)) + ...
           offset * elements(nindex) + 1;
   stop = start + elements(nindex) - 1;
   ntst = n;
otherwise
   error('TYPE must begin with "a", "h", "v", or "d".');
end

switch lower(opcode)                   % Do requested action.
case {'copy', 'cut'}
   y = repmat(0, s(nindex, :));
   y(:) = c(start:stop);    nc = c;
   if strcmp(lower(opcode(1:3)), 'cut')
      nc(start: stop) = 0; varargout = {nc, y};
   else 
      varargout = {y};    
   end
case 'paste'
   if prod(size(x)) ~= elements(end - ntst)
      error('X is not sized for the requested paste.');
   else
      nc = c;   nc(start:stop) = x(:);   varargout = {nc};  
   end
otherwise
   error('Unrecognized OPCODE.');
end

