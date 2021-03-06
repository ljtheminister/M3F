function [model] = m3f_tib_initModel(numUsers, numItems, numFacs, KU, KM, W0,...
                                       nu0, mu0, lambda0, alpha, sigmaSqd, ...
                                       sigmaSqd0, c0, d0, chi0)
%M3F_TIB_INITMODEL Initialize and return a TIB model structure.
%
% Usage:
%    [model] = m3f_tib_initModel(numUsers, numItems, numFacs, KU, KM)
%    [model] = m3f_tib_initModel(numUsers, numItems, numFacs, KU, KM, W0,
%              nu0, mu0, lambda0, alpha, sigmaSqd, sigmaSqd0, c0, d0, chi0)
%
% Inputs:
%    numUsers - Maximum user id
%    numItems - Maximum item id
%    numFacs - Number of static latent factors
%    KU - Number of user topics
%    KM - Number of item topics
%    See Mackey et al. for definitions of remaining free parameters:
%    W0 (numFacs x numFacs)
%    nu0 (1 x 1)
%    mu0 (numFacs x 1)
%    lambda0 (1 x 1)
%    alpha (1 x 1)
%    sigmaSqd (1 x 1)
%    sigmaSqd0 (1 x 1)
%    c0 (1 x 1)
%    d0 (1 x 1)
%    chi0 (1 x 1)
%
% Outputs:
%    model - TIB model structure initialized according to
%            given inputs and default values for unspecified free
%            parameters

% -----------------------------------------------------------------------     
%
% Last revision: 2-July-2010
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

if nargin < 6
   model.numUsers = numUsers;
   model.numItems = numItems;
   model.KU = KU;
   model.KM = KM;
   
   % Default values for free parameters
   model.W0 = eye(numFacs);
   model.nu0 = numFacs;
   model.mu0 = zeros(numFacs,1);
   model.lambda0 = 10; 
   model.alpha = 10000;
   model.sigmaSqd = .5;
   model.sigmaSqd0 = .1;
   model.c0 = 0;
   model.d0 = 0;
   model.chi0 = 0;
else
   model.numUsers = numUsers;
   model.numItems = numItems;
   model.KU = KU;
   model.KM = KM;
   model.W0 = W0;
   model.nu0 = nu0;
   model.mu0 = mu0;
   model.lambda0 = lambda0;
   model.alpha = alpha;
   model.sigmaSqd = sigmaSqd;
   model.sigmaSqd0 = sigmaSqd0;
   model.c0 = c0;
   model.d0 = d0;
   model.chi0 = chi0;
end

% -----------------------------END OF CODE-------------------------------
