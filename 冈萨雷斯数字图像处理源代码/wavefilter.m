function [varargout] = wavefilter(wname, type)
%WAVEFILTER Create wavelet decomposition and reconstruction filters.
%   [VARARGOUT] = WAVEFILTER(WNAME, TYPE) returns the decomposition
%   and/or reconstruction filters used in the computation of the
%   forward and inverse FWT (fast wavelet transform). 
%
%   EXAMPLES:
%     [ld, hd, lr, hr] = wavefilter('haar') Get the low and highpass 
%                                           decomposition (ld, hd) 
%                                           and reconstruction 
%                                           (lr, hr) filters for 
%                                           wavelet 'haar'.
%     [ld, hd] = wavefilter('haar','d')     Get decomposition filters
%                                           ld and hd.
%     [lr, hr] = wavefilter('haar','r')     Get reconstruction 
%                                           filters lr and hr.
%
%   INPUTS:
%     WNAME             Wavelet Name
%     ---------------------------------------------------------
%     'haar' or 'db1'   Haar
%     'db4'             4th order Daubechies
%     'sym4'            4th order Symlets
%     'bior6.8'         Cohen-Daubechies-Feauveau biorthogonal
%     'jpeg9.7'         Antonini-Barlaud-Mathieu-Daubechies
%
%     TYPE              Filter Type
%     ---------------------------------------------------------
%     'd'               Decomposition filters
%     'r'               Reconstruction filters
%
%   See also WAVEFAST and WAVEBACK.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2003/10/13 01:09:39 $

% Check the input and output arguments.
error(nargchk(1, 2, nargin));

if (nargin == 1 & nargout ~= 4) | (nargin == 2 & nargout ~= 2)
   error('Invalid number of output arguments.'); 
end

if nargin == 1 & ~ischar(wname)
   error('WNAME must be a string.'); 
end

if nargin == 2 & ~ischar(type)
   error('TYPE must be a string.'); 
end
  
% Create filters for the requested wavelet.
switch lower(wname)
case {'haar', 'db1'}
   ld = [1 1]/sqrt(2);     hd = [-1 1]/sqrt(2);
   lr = ld;                hr = -hd;
   
case 'db4'
   ld = [-1.059740178499728e-002 3.288301166698295e-002 ...
         3.084138183598697e-002 -1.870348117188811e-001 ...
         -2.798376941698385e-002 6.308807679295904e-001 ...
         7.148465705525415e-001 2.303778133088552e-001];
   t = (0:7);
   hd = ld;    hd(end:-1:1) = cos(pi * t) .* ld;
   lr = ld;    lr(end:-1:1) = ld;
   hr = cos(pi * t) .* ld;
   
case 'sym4'
   ld = [-7.576571478927333e-002 -2.963552764599851e-002 ...
         4.976186676320155e-001 8.037387518059161e-001 ...
         2.978577956052774e-001 -9.921954357684722e-002 ...
         -1.260396726203783e-002 3.222310060404270e-002];
   t = (0:7);
   hd = ld;    hd(end:-1:1) = cos(pi * t) .* ld;
   lr = ld;    lr(end:-1:1) = ld;
   hr = cos(pi * t) .* ld;
   
case 'bior6.8'
   ld = [0 1.908831736481291e-003 -1.914286129088767e-003 ...
         -1.699063986760234e-002 1.193456527972926e-002 ...
         4.973290349094079e-002 -7.726317316720414e-002 ...
         -9.405920349573646e-002 4.207962846098268e-001 ...
         8.259229974584023e-001 4.207962846098268e-001 ...
         -9.405920349573646e-002 -7.726317316720414e-002 ...
         4.973290349094079e-002 1.193456527972926e-002 ...
         -1.699063986760234e-002 -1.914286129088767e-003 ...
         1.908831736481291e-003];
   hd = [0 0 0 1.442628250562444e-002 -1.446750489679015e-002 ...
         -7.872200106262882e-002 4.036797903033992e-002 ...
         4.178491091502746e-001 -7.589077294536542e-001 ...
         4.178491091502746e-001 4.036797903033992e-002 ...
         -7.872200106262882e-002 -1.446750489679015e-002 ...
         1.442628250562444e-002 0 0 0 0];
   t = (0:17);
   lr = cos(pi * (t + 1)) .* hd;
   hr = cos(pi * t) .* ld;
   
case 'jpeg9.7'
   ld = [0 0.02674875741080976 -0.01686411844287495 ...
         -0.07822326652898785 0.2668641184428723 ...
         0.6029490182363579 0.2668641184428723 ...
         -0.07822326652898785 -0.01686411844287495 ...
         0.02674875741080976];
   hd = [0 -0.09127176311424948 0.05754352622849957 ...
         0.5912717631142470 -1.115087052456994 ...
         0.5912717631142470 0.05754352622849957 ...
         -0.09127176311424948 0 0];
   t = (0:9);
   lr = cos(pi * (t + 1)) .* hd;
   hr = cos(pi * t) .* ld;
   
otherwise
   error('Unrecognizable wavelet name (WNAME).');
end

% Output the requested filters.
if (nargin == 1)
   varargout(1:4) = {ld, hd, lr, hr};
else
   switch lower(type(1))
   case 'd'
      varargout = {ld, hd};
   case 'r'
      varargout = {lr, hr};
   otherwise
      error('Unrecognizable filter TYPE.');
   end
end
