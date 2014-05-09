
% -----------------------------------------------------------------------     
%
% Last revision: 9-July-2010
%
% Authors: Lester Mackey and David Weiss
% License: MIT License
%
% Copyright (c) 2010 Lester Mackey & David Weiss
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to
% permit persons to whom the Software is furnished to do so, subject to
% the following conditions:
% 
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
% OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% -----------------------------------------------------------------------

% -----------------------------BEGIN CODE--------------------------------

test_tib = true;
test_tif = true;

if test_tib
    mkdir('testm3f_tib');
    mkdir('testm3f_tib/models');
    mkdir('testm3f_tib/samples');
    mkdir('testm3f_tib/log');
    
    err = m3f_tib_exper('testm3f_tib', 'movielens100k', [1 2 3], 3, ...
        12345, 10, 2, 2);
end

if test_tif
    mkdir('testm3f_tif');
    mkdir('testm3f_tif/models');
    mkdir('testm3f_tif/samples');
    mkdir('testm3f_tif/log');
    
    err = m3f_tif_exper('testm3f_tif', 'movielens100k', [1 2 3], 3, ...
        12345, 10, 2, 2, 2);
end
%% 

% -----------------------------END OF CODE-------------------------------
