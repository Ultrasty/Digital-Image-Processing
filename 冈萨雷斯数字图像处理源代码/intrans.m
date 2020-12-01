function g = intrans(f, varargin)
%INTRANS Performs intensity (gray-level) transformations.
%   G = INTRANS(F, 'neg') computes the negative of input image F.
% 
%   G = INTRANS(F, 'log', C, CLASS) computes C*log(1 + F) and
%   multiplies the result by (positive) constant C. If the last two
%   parameters are omitted, C defaults to 1. Because the log is used
%   frequently to display Fourier spectra, parameter CLASS offers the
%   option to specify the class of the output as 'uint8' or
%   'uint16'. If parameter CLASS is omitted, the output is of the 
%   same class as the input. 
% 
%   G = INTRANS(F, 'gamma', GAM) performs a gamma transformation on
%   the input image using parameter GAM (a required input).  
%
%   G = INTRANS(F, 'stretch', M, E) computes a contrast-stretching
%   transformation using the expression 1./(1 + (M./(F +
%   eps)).^E).  Parameter M must be in the range [0, 1].  The default
%   value for M is mean2(im2double(F)), and the default value for E
%   is 4.
%
%   For the 'neg', 'gamma', and 'stretch' transformations, double
%   input images whose maximum value is greater than 1 are scaled
%   first using MAT2GRAY.  Other images are converted to double first
%   using IM2DOUBLE.  For the 'log' transformation, double images are
%   transformed without being scaled; other images are converted to
%   double first using IM2DOUBLE.
%
%   The output is of the same class as the input, except if a
%   different class is specified for the 'log' option.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.7 $  $Date: 2003/10/13 00:45:53 $

% Verify the correct number of inputs.
error(nargchk(2, 4, nargin))

% Store the class of the input for use later.
classin = class(f);

% If the input is of class double, and it is outside the range
% [0, 1], and the specified transformation is not 'log', convert the
% input to the range [0, 1].
if strcmp(class(f), 'double') & max(f(:)) > 1 & ...
      ~strcmp(varargin{1}, 'log')
   f = mat2gray(f);
else % Convert to double, regardless of class(f).
   f = im2double(f);
end

% Determine the type of transformation specified.
method = varargin{1};

% Perform the intensity transformation specified.    
switch method
case 'neg' 
   g = imcomplement(f); 

case 'log'
   if length(varargin) == 1  
      c = 1;
   elseif length(varargin) == 2  
      c = varargin{2}; 
   elseif length(varargin) == 3 
      c = varargin{2}; 
      classin = varargin{3};
   else 
      error('Incorrect number of inputs for the log option.')
   end
   g = c*(log(1 + double(f)));

case 'gamma'
   if length(varargin) < 2
      error('Not enough inputs for the gamma option.')
   end
   gam = varargin{2}; 
   g = imadjust(f, [ ], [ ], gam);
   
case 'stretch'
   if length(varargin) == 1
      % Use defaults.
      m = mean2(f);  
      E = 4.0;           
   elseif length(varargin) == 3
      m = varargin{2};  
      E = varargin{3};
   else error('Incorrect number of inputs for the stretch option.')
   end
   g = 1./(1 + (m./(f + eps)).^E);
otherwise
   error('Unknown enhancement method.')
end

% Convert to the class of the input image.
g = changeclass(classin, g);
