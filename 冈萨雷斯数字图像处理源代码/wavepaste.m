function nc = wavepaste(type, c, s, n, x)
%WAVEPASTE Puts coefficients in a wavelet decomposition structure.
%   NC = WAVEPASTE(TYPE, C, S, N, X) returns the new decomposition
%   structure after pasting X into it based on TYPE and N.
%
%   INPUTS:
%     TYPE      Coefficient category
%     -------------------------------------
%     'a'       Approximation coefficients
%     'h'       Horizontal details
%     'v'       Vertical details
%     'd'       Diagonal details
%
%     [C, S] is a wavelet data structure.
%     N specifies a decomposition level (Ignored if TYPE = 'a').
%     X is a two-dimensional approximation or detail coefficient
%       matrix whose dimensions are appropriate for decomposition
%       level N.
%
%   See also WAVEWORK, WAVECUT, and WAVECOPY.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.4 $  $Date: 2003/10/13 01:21:13 $

error(nargchk(5, 5, nargin))
nc = wavework('paste', type, c, s, n, x);
