function x = huff2mat(y)
%HUFF2MAT Decodes a Huffman encoded matrix.
%   X = HUFF2MAT(Y) decodes a Huffman encoded structure Y with uint16
%   fields: 
%      Y.min    Minimum value of X plus 32768
%      Y.size   Size of X
%      Y.hist   Histogram of X
%      Y.code   Huffman code
%
%   The output X is of class double.
%
%   See also MAT2HUFF.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2003/11/21 13:17:50 $

if ~isstruct(y) | ~isfield(y, 'min') | ~isfield(y, 'size') | ...
       ~isfield(y, 'hist') | ~isfield(y, 'code')
   error('The input must be a structure as returned by MAT2HUFF.');
end

sz = double(y.size);   m = sz(1);   n = sz(2);
xmin = double(y.min) - 32768;            % Get X minimum
map = huffman(double(y.hist));           % Get Huffman code (cell)

% Create a binary search table for the Huffman decoding process.
% 'code' contains source symbol strings corresponding to 'link'
% nodes, while 'link' contains the addresses (+) to node pairs for
% node symbol strings plus '0' and '1' or addresses (-) to decoded
% Huffman codewords in 'map'. Array 'left' is a list of nodes yet to
% be processed for 'link' entries.

code = cellstr(char('', '0', '1'));     % Set starting conditions as
link = [2; 0; 0];   left = [2 3];       % 3 nodes w/2 unprocessed
found = 0;   tofind = length(map);      % Tracking variables

while length(left) & (found < tofind)
   look = find(strcmp(map, code{left(1)}));    % Is string in map?
   if look                            % Yes
      link(left(1)) = -look;          % Point to Huffman map
      left = left(2:end);             % Delete current node
      found = found + 1;              % Increment codes found
      
   else                               % No, add 2 nodes & pointers
      len = length(code);             % Put pointers in node
      link(left(1)) = len + 1;
      
      link = [link; 0; 0];            % Add unprocessed nodes
      code{end + 1} = strcat(code{left(1)}, '0');
      code{end + 1} = strcat(code{left(1)}, '1');
      
      left = left(2:end);             % Remove processed node
      left = [left len + 1 len + 2];  % Add 2 unprocessed nodes
   end
end

x = unravel(y.code', link, m * n);    % Decode using C 'unravel'
x = x + xmin - 1;                     % X minimum offset adjust
x = reshape(x, m, n);                 % Make vector an array
