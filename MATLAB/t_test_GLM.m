%==========================================================================
% Function: t_test_GLM
%
% Description:
%   Performs a general linear model (GLM) t-test for a specified contrast.
%   It calculates the t-statistic and its corresponding p-value under
%   the null hypothesis c'*beta = d.
%
% Syntax:
%   [t, p_val] = t_test_GLM(X, y, c)
%   [t, p_val] = t_test_GLM(X, y, c, d)
%
% Inputs:
%   X : [J x p] design matrix (predictors)
%   y : [J x 1] response vector (observations)
%   c : [p x 1] contrast vector
%   d : scalar (optional), null hypothesis value (default = 0)
%
% Outputs:
%   t     : t-statistic for the contrast
%   p_val : two-tailed p-value associated with the t-statistic
%
% Example:
%   X = [1 1; 1 2; 1 3];
%   y = [1; 2; 2.5];
%   c = [0; 1];
%   [t, p_val] = t_test_GLM(X, y, c)
%
% Author:
%   Youngjo Song
%
% Date:
%   May 26, 2025
%==========================================================================

function [t, p_val] = t_test_GLM(X, y, c, d)
    % Set default value of d to 0 if not provided
    if nargin < 4
        d = 0;
    end

    % Number of observations (rows in X)
    J = size(X, 1);

    % Rank of the design matrix (number of independent columns)
    p = rank(X);

    % Estimate beta using Moore-Penrose pseudoinverse
    beta = pinv(X) * y;

    % Compute residuals
    resid_e = y - X * beta;

    % Estimate residual variance (sigma^2)
    sigma_squared = (resid_e' * resid_e) / (J - p);

    % Compute t-statistic for the contrast
    t = (c' * beta - d) / sqrt(sigma_squared * c' * pinv(X' * X) * c);

    % Compute two-tailed p-value from the t-distribution
    p_val = 2 * (1 - tcdf(abs(t), J - p));
end
