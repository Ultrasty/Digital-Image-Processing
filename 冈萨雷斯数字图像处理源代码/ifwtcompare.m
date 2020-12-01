function [ratio, maxdiff] = ifwtcompare(f, n, wname)
%IFWTCOMPARE Compare waverec2 and waveback.
%   [RATIO, MAXDIFF] = IFWTCOMPARE(F, N, WNAME) compares the
%   operation of Wavelet Toolbox function WAVEREC2 and custom
%   function WAVEBACK. 
%
%   INPUTS:
%     F           Image to transform and inverse transform.
%     N           Number of scales to compute.
%     WNAME       Wavelet to use.
%
%   OUTPUTS:
%     RATIO       Execution time ratio (custom/toolbox).
%     MAXDIFF     Maximum generated image difference.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2003/10/13 01:30:46 $

% Compute the transform and get output and computation time for
% waverec2. 
[c1, s1] = wavedec2(f, n, wname);
tic;
g1 = waverec2(c1, s1, wname);
reftime = toc;

% Compute the transform and get output and computation time for
% waveback. 
[c2, s2] = wavefast(f, n, wname);
tic;
g2 = waveback(c2, s2, wname);
t2 = toc;

% Compare the results.
ratio = t2 / (reftime + eps);
maxdiff = abs(max(max(g1 - g2)));
