function g = splitmerge(f, mindim, fun)
%SPLITMERGE Segment an image using a split-and-merge algorithm.
%   G = SPLITMERGE(F, MINDIM, @PREDICATE) segments image F by using a
%   split-and-merge approach based on quadtree decomposition. MINDIM
%   (a positive integer power of 2) specifies the minimum dimension
%   of the quadtree regions (subimages) allowed. If necessary, the
%   program pads the input image with zeros to the nearest square  
%   size that is an integer power of 2. This guarantees that the  
%   algorithm used in the quadtree decomposition will be able to 
%   split the image down to blocks of size 1-by-1. The result is  
%   cropped back to the original size of the input image. In the  
%   output, G, each connected region is labeled with a different
%   integer.
%
%   Note that in the function call we use @PREDICATE for the value of 
%   fun.  PREDICATE is a function in the MATLAB path, provided by the
%   user. Its syntax is
%
%       FLAG = PREDICATE(REGION) which must return TRUE if the pixels
%       in REGION satisfy the predicate defined by the code in the
%       function; otherwise, the value of FLAG must be FALSE.
% 
%   The following simple example of function PREDICATE is used in 
%   Example 10.9 of the book.  It sets FLAG to TRUE if the 
%   intensities of the pixels in REGION have a standard deviation  
%   that exceeds 10, and their mean intensity is between 0 and 125. 
%   Otherwise FLAG is set to false. 
%
%       function flag = predicate(region)
%       sd = std2(region);
%       m = mean2(region);
%       flag = (sd > 10) & (m > 0) & (m < 125);

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.6 $  $Date: 2003/10/26 22:36:01 $

% Pad image with zeros to guarantee that function qtdecomp will
% split regions down to size 1-by-1.
Q = 2^nextpow2(max(size(f)));
[M, N] = size(f);
f = padarray(f, [Q - M, Q - N], 'post');

%Perform splitting first. 
S = qtdecomp(f, @split_test, mindim, fun);

% Now merge by looking at each quadregion and setting all its 
% elements to 1 if the block satisfies the predicate.

% Get the size of the largest block. Use full because S is sparse.
Lmax = full(max(S(:)));
% Set the output image initially to all zeros.  The MARKER array is
% used later to establish connectivity.
g = zeros(size(f));
MARKER = zeros(size(f));
% Begin the merging stage.
for K = 1:Lmax 
   [vals, r, c] = qtgetblk(f, S, K);
   if ~isempty(vals)
      % Check the predicate for each of the regions
      % of size K-by-K with coordinates given by vectors
      % r and c.
      for I = 1:length(r)
         xlow = r(I); ylow = c(I);
         xhigh = xlow + K - 1; yhigh = ylow + K - 1;
         region = f(xlow:xhigh, ylow:yhigh);
         flag = feval(fun, region);
         if flag 
            g(xlow:xhigh, ylow:yhigh) = 1;
            MARKER(xlow, ylow) = 1;
         end
      end
   end
end

% Finally, obtain each connected region and label it with a
% different integer value using function bwlabel.
g = bwlabel(imreconstruct(MARKER, g));

% Crop and exit
g = g(1:M, 1:N);

%-------------------------------------------------------------------%
function v = split_test(B, mindim, fun)
% THIS FUNCTION IS PART OF FUNCTION SPLIT-MERGE. IT DETERMINES 
% WHETHER QUADREGIONS ARE SPLIT. The function returns in v 
% logical 1s (TRUE) for the blocks that should be split and 
% logical 0s (FALSE) for those that should not.

% Quadregion B, passed by qtdecomp, is the current decomposition of
% the image into k blocks of size m-by-m.

% k is the number of regions in B at this point in the procedure.
k = size(B, 3);

% Perform the split test on each block. If the predicate function
% (fun) returns TRUE, the region is split, so we set the appropriate
% element of v to TRUE. Else, the appropriate element of v is set to
% FALSE.
v(1:k) = false;
for I = 1:k
   quadregion = B(:, :, I);
   if size(quadregion, 1) <= mindim
      v(I) = false;
      continue
   end
   flag = feval(fun, quadregion);
   if flag
      v(I) = true;
   end
end

