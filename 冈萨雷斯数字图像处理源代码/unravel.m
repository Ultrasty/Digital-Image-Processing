%UNRAVEL Decodes a variable-length bit stream.
%   X = UNRAVEL(Y, LINK, XLEN) decodes UINT16 input vector Y based on
%   transition and output table LINK. The elements of Y are
%   considered to be a contiguous stream of encoded bits--i.e., the
%   MSB of one element follows the LSB of the previous element. Input
%   XLEN is the number code words in Y, and thus the size of output
%   vector X (class DOUBLE). Input LINK is a transition and output
%   table (that drives a series of binary searches):
%
%    1. LINK(0) is the entry point for decoding, i.e., state n = 0.
%    2. If LINK(n) < 0, the decoded output is |LINK(n)|; set n = 0.
%    3. If LINK(n) > 0, get the next encoded bit and transition to
%       state [LINK(n) - 1] if the bit is 0, else LINK(n).

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.4 $  $Date: 2003/10/26 18:41:41 $
