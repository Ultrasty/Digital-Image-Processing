function image = changeclass(class, varargin)
%CHANGECLASS changes the storage class of an image.
%  I2 = CHANGECLASS(CLASS, I);
%  RGB2 = CHANGECLASS(CLASS, RGB);
%  BW2 = CHANGECLASS(CLASS, BW);
%  X2 = CHANGECLASS(CLASS, X, 'indexed');

%  Copyright 1993-2002 The MathWorks, Inc.  Used with permission.
%  $Revision: 1.2 $  $Date: 2003/02/19 22:09:58 $

switch class
case 'uint8'
   image = im2uint8(varargin{:});
case 'uint16'
   image = im2uint16(varargin{:});
case 'double'
   image = im2double(varargin{:});
otherwise
   error('Unsupported IPT data class.');
end

