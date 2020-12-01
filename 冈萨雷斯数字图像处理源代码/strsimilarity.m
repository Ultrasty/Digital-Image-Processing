function R = strsimilarity(a, b)
%STRSIMILARITY Computes a similarity measure between two strings.
%   R = STRSIMILARITY(A, B) computes the similarity measure, R,
%   defined in Section 12.4.2 for strings A and B. The strings do not
%   have to be of the same length, but if one is shorter than other,
%   then it is assumed that the shorter string has been padded with
%   leading blanks so that it is brought into the necessary
%   registration prior to using this function. Only one of the
%   strings can have blanks, and these must be leading and/or
%   trailing blanks. Blanks are not counted when computing the length
%   of the strings for use in the similarity measure. 

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.4 $  $Date: 2003/05/18 18:07:48 $

% Verify that a and b are character strings.
if ~ischar(a) | ~ischar(b)
   error('Inputs must be character strings.')
end

% Find any blank spaces.
I = find(a == ' ');
J = find(b == ' ');
LI = length(I); LJ = length(J);
if LI ~= 0 & LJ ~= 0
   error('Only one of the strings can contain blanks.')
end

% Pad the end of the appropriate string. It is assumed
% that they are registered in terms of their beginning
% positions.
a = a(:); b = b(:);
La = length(a); Lb = length(b);
if LI == 0 & LJ == 0
   if La > Lb
      b = [b; blanks(La - Lb)'];
   else 
      a = [a; blanks(Lb - La)'];
   end
elseif isempty(I)
   Lb = length(b) - length(J);
   b = [b; blanks(La - Lb - LJ)'];
else
   La = length(a) - length(I);
   a = [a; blanks(Lb - La - LI)'];
end

% Compute the similarity measure.
I = find(a == b);
alpha = length(I);
den = max(La, Lb) - alpha;
if den == 0 
   R = Inf;
else 
   R = alpha/den; 
end
