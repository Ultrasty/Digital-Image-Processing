% DIPUM Toolbox.
% Version 1.1.4  15-July-2006
%
% M-files from the book Digital Image Processing Using MATLAB, by
% R.C. Gonzalez, R.E. Woods, and S.L. Eddins, Prentice Hall, 2004.
% http://www.prenhall.com/gonzalezwoodseddins
%
% Release information.
%   Readme        - Information about current and previous versions.
%
% Image display.
%   ice           - Interactive color editing.
%   rgbcube       - Display a color RGB cube.
%
% Geometric transformations.
%   pixeldup      - Duplicate pixels of an image in both directions.
%   vistformfwd   - Visualize forward geometric transformation.
%
% Pixel values and statistics.
%   covmatrix     - Covariance matrix of vector population.
%   statmoments   - Statistical central moments of image histogram.
%
% Image analysis, including segmentation, description, recognition.
%   bayesgauss    - Bayes classifier for Gaussian patterns.
%   bound2eight   - Convert 4-connected boundary to 8-connected.
%   bound2four    - Convert 8-connected boundary to 4-connected.
%   bound2im      - Convert boundary to image.
%   boundaries    - Trace region boundaries.
%   bsubsamp      - Subsample a boundary.
%   colorgrad     - Vector gradient of RGB image.
%   colorseg      - Segment a color image.
%   connectpoly   - Connect vertices of polygon.
%   diameter      - Measure diameter of image regions.
%   fchcode       - Freeman chain code of boundary.
%   frdescp       - Fourier descriptors.
%   hough         - Hough transform.
%   houghlines    - Extract line segments using Hough transform.
%   houghpeaks    - Detect peaks in Hough transform.
%   houghpixels   - Image pixels associated with Hough transform bin.
%   ifrdescp      - Inverse Fourier descriptors.
%   imstack2vectors - Extract vectors from image stack.
%   invmoments    - Invariant moments of image.
%   mahalanobis   - Mahalanobis distance.
%   minperpoly    - Minimum perimeter polygon.
%   polyangles    - Internal angles of polygon vertices.
%   princomp      - Principal-component vectors.
%   randvertex    - Randomly perturb polygon vertices.
%   regiongrow    - Segmentation by region growing.
%   signature     - Signature of boundary.
%   specxture     - Spectral texture of image.
%   splitmerge    - Segment image using split-and-merge.
%   statxture     - Statistical texture measures of image.
%   strsimilarity - Similarity measurement between two strings.
%   x2majoraxis   - Align coordinate x with major axis of region.
%
% Image compression.
%   compare       - Compute and display error between two matrices.
%   entropy       - Estimate entropy of a matrix.
%   huff2mat      - Decode a Huffman-encoded matrix.
%   huffman       - Build Huffman code for symbol source.
%   im2jpeg       - Compress image using JPEG approximation.
%   im2jpeg2k     - Compress image using JPEG 2000 approximation.
%   imratio       - Byte ratio for two image files or variables.
%   jpeg2im       - Decompress an IM2JPEG-compressed image.
%   jpeg2k2im     - Decompress an IM2JPEG2K-compressed image.
%   lpc2mat       - Decompress lossless predictive encoded matrix.
%   mat2huff      - Huffman-encode a matrix.
%   mat2lpc       - Compress matrix using lossless predictive coding.
%   quantize      - Quantize elements of uint8 matrix.
%
% Image enhancement.
%   gscale        - Scale intensity values of image.
%   intrans       - Perform several intensity transformations.
%
% Image noise.
%   imnoise2      - Generate random array using specified PDF.
%   imnoise3      - Generate periodic noise.
%
% Linear and nonlinear spatial filtering.
%   adpmedian     - Adaptive median filtering.
%   dftcorr       - Frequency-domain correlation.
%   dftfilt       - Frequency-domain filtering.
%   spfilt        - Linear and nonlinear spatial filtering.
%
% Linear 2-D filter design.
%   hpfilter      - Frequency-domain highpass filters.
%   lpfilter      - Frequency-domain lowpass filters.
%
% Wavelets.
%   wave2gray     - Display wavelet decomposition coefficients.
%   waveback      - Multilevel 2-D inverse fast wavelet transform.
%   wavecopy      - Fetch coefficients of wavelet decomposition.
%   wavecut       - Set to zero coefficients of wavelet decomposition.
%   wavefast      - Multilevel 2-D fast wavelet transform.
%   wavefilter    - Wavelet decomposition and reconstruction filters.
%   wavepaste     - Put coefficients into wavelet decomposition.
%   wavework      - Edit wavelet decomposition structure.
%   wavezero      - Set wavelet detail coefficients to zero.
%
% Morphological operations.
%   endpoints     - End-points of binary image.
%
% Region-based processing.
%   histroi       - Histogram of ROI in an image.
%
% Color space conversions.
%   hsi2rgb       - Convert HSI values to RGB color space.
%   rgb2hsi       - Convert RGB values to HSI color space.
%
% Array operations.
%   dftuv         - Mesgrid frequency arrays.
%   paddedsize    - Recommended pad size for FFT-based filtering.
%
% Miscellaneous.
%   conwaylaws    - Apply Conway's genetic laws to a pixel.
%   manualhist    - Generate 2-mode histogram interactively.
%   twomodegauss  - Generate 2-mode Gaussian function.
%   unravel       - Utility MEX-file used for Huffman decoding.
%
% Functions used only for illustration in the book.
%   average       - Average value of array.
%   fwtcompare    - Compare DIPUM and Wavelet Toolbox FWT functions.
%   gmean         - Geometric mean of columns.
%   ifwtcompare   - Compare DIPUM and Wavelet Toolbox IFWT functions.
%   improd        - Product of two images.
%   subim         - Extract subimage.
%   twodsin       - Compare loop vs. vectorized implementation.
%
% Undocumented IPT functions (used here with permission).
%   changeclass   - Scale image and change its class.
%   intline       - Integer-coordinate line-drawing algorithm.

%   Copyright 2002-2006 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.9 $  $Date: 2006/07/15 20:43:27 $
