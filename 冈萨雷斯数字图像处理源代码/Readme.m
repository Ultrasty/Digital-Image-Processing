% README file for the DIPUM Toolbox.
%
% These MATLAB functions were created for the book Digital Image
% Processing Using MATLAB (DIPUM), by R.C. Gonzalez, R.E. Woods, and
% S.L. Eddins, Prentice Hall, 2004, ISBN 0130085197.  The functions
% supplement and extend the image processing functionality of MATLAB
% and the Image Processing Toolbox, both offered by The MathWorks.
%
% The free version of the DIPUM Toolbox
% -------------------------------------
% The free version of the DIPUM Toolbox contains only executable
% MATLAB P-files.  P-files may be executed as MATLAB functions just
% as M-files.  Unlike M-files, however, neither the help text nor the
% source code is accessible.  Complete source code listings for all
% functions are included in the book.
%
% The registered version of the DIPUM Toolbox
% -------------------------------------------
% The registered version of the DIPUM Toolbox contains M-files with
% readable source code and help text.  Users who want the M-files in
% electronic form may obtain, for a small fee, a password allowing them
% to download the registered toolbox.  More information about obtaining
% the registered toolbox version is available at the book's web site:
%
%     http://www.imageprocessingbook.com/index_dipum.htm
%
% Requirements
% ------------
% This initial release of the DIPUM Toolbox was developed and tested
% using MATLAB 6.5 and the Image Processing Toolbox 3.1.  The DIPUM
% Toolbox is not supported for use with earlier versions of MATLAB or
% the Image Processing Toolbox.  Most DIPUM functions require the
% Image Processing Toolbox.   
%
% The DIPUM Toolbox works with all MATLAB platforms.  It includes one
% MEX-file (UNRAVEL, which is used for Huffman decoding).  Compiled
% binaries for this MEX-file are provided for all MATLAB platforms.
%
% Installation
% ------------
% To install the DIPUM Toolbox, place the folder containing the
% functions on a local hard drive or a network drive accessible to
% your computer, and then include the folder location in the MATLAB
% path.  To set the MATLAB path, start MATLAB and then select the
% File/Set Path menu item.  Then select Add Folder.  Use the
% navigation window to select the folder containing the functions.
% Click OK and then click Save.  The functions will then be ready for
% use within MATLAB.
%
% Copyright
% ---------
% The DIPUM Toolbox is protected under U.S. and international copyright
% law.  You may not transfer, publish, disclose, display, or otherwise
% make available either the free or the registered versions of the DIPUM
% Toolbox to other people. In particular, you may not place either
% version of the DIPUM Toolbox on an Internet site, a computer bulletin
% board, or an anonymous ftp directory from which it could be downloaded
% or copied by other people.  Anyone may obtain the free version
% directly from the book's web site.
%
% Disclaimer of Warranty
% ----------------------
% The DIPUM Toolbox source code is furnished "as is." Neither the
% authors nor the publisher of Digital Image Processing Using MATLAB
% make or imply any warranties that the DIPUM Toolbox software is free
% of error, that it is consistent with any particular standard of
% merchantability, or that it will meet your requirements for any
% particular application. The authors accept no responsibility for any
% mathematical or technical limitations of the functions that make up
% the DIPUM Toolbox. The DIPUM Toolbox should not be relied upon for
% solving a problem whose incorrect solution could result in injury to a
% person or loss of property. Neither the authors nor the publisher of
% Digital Image Processing Using MATLAB shall in any event be liable for
% any damages, whether direct or indirect, special or general,
% consequential or incidental, arising from use of the DIPUM Toolbox.
%
% The DIPUM Toolbox was not created by The MathWorks, nor is it
% supported by The MathWorks.
%
% About Version 1.1.4
% -------------------
% Release date: 15-July-2006
%
% Changes:
%   * bayesgauss:  Corrected typo in help text.
%   * ifrdescp:    Corrected typo in help text.
%   * im2jpeg:     Corrected bug in end-of-block computation.
%   * imnoise2:    Corrected bug in the lognormal computation.
%   * mahalanobis: Corrected typo in help text.
%   * pixeldup:    Corrected typo in comment.
%   * rgb2hsi:     Corrected typo in comment.
%
% About Version 1.1.3
% -------------------
% Release date: 15-Dec-2004
%
% Changes:
%   * bayesgauss: Corrected bug in the optimized version that was
%                 released in DIPUM Toolbox 1.1.
%
% About Version 1.1.2
% -------------------
% Release date: 04-Nov-2004
%
% Changes:
%   * bayesgauss: Revised help text and some comments in the code.
%
%   * bsubsamp:   Corrected indexing errors that occurred under some 
%                 conditions.
%
%   * ifrdescp:   Revised help text.
%
%   * imnoise3:   Revised help text.
%
%   * lpfilter:   Removed a superfluous output variable.
%
%   * statxture:  Removed unnecessary brackets surrounding output
%                 variable.
%
% About Version 1.1.1
% -------------------
% Release date: 04-Jun-2004
%
% Changes:
%   * Added missing ice.fig file to toolbox distribution.
%
% About Version 1.1
% -----------------
% Release date: 10-Mar-2004
%
% Changes:
%   * Function bayesgauss was modified to improve performance.
%
% About Version 1.0
% -----------------
% Release date: 01-Dec-2003
%
% This initial release of the DIPUM Toolbox coincides with the
% initial publication of Digital Image Processing Using MATLAB.
%
% Copyright 2002-2006 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
% Digital Image Processing Using MATLAB, Prentice-Hall, 2004
% $Revision: 1.15 $  $Date: 2006/07/15 20:43:48 $
