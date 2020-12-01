function [ratio, maxdiff] = fwtcompare(f, n, wname)
%FWTCOMPARE Compare wavedec2 and wavefast.
%   [RATIO, MAXDIFF] = FWTCOMPARE(F, N, WNAME) compares the operation
%   of toolbox function WAVEDEC2 and custom function WAVEFAST. 
%
%   INPUTS:
%     F           Image to be transformed.
%     N           Number of scales to compute.
%     WNAME       Wavelet to use.
%
%   OUTPUTS:
%     RATIO       Execution time ratio (custom/toolbox)
%     MAXDIFF     Maximum coefficient difference.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.3 $  $Date: 2003/04/18 05:07:33 $

% Get transform and computation time for wavedec2.
tic;
[c1, s1] = wavedec2(f, n, wname);
reftime = toc;

% Get transform and computation time for wavefast.
tic;
[c2, s2] = wavefast(f, n, wname);
t2 = toc;

% Compare the results.
ratio = t2 / (reftime + eps);
maxdiff = abs(max(c1 - c2));
