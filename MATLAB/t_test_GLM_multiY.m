%==========================================================================
% Function: t_test_GLM_multiY
%
% Description:
%   Performs GLM t-tests for multiple response variables simultaneously.
%   Each column of Y is treated as an independent response vector.
%
% Inputs:
%   X : [J x p] design matrix
%   Y : [J x K] response matrix (K responses)
%   c : [p x 1] contrast vector
%   d : scalar (optional), null hypothesis value (default = 0)
%
% Outputs:
%   t     : [1 x K] t-statistics
%   p_val : [1 x K] two-tailed p-values
%==========================================================================

function [t, p_val] = t_test_GLM_multiY(X, Y, c, d)

    if nargin < 4
        d = 0;
    end

    % Dimensions
    J = size(X, 1);
    p = rank(X);

    % Pseudoinverse of X
    X_pinv = pinv(X);

    % Beta estimates for all Y columns (p x K)
    beta = X_pinv * Y;

    % Residuals (J x K)
    resid_e = Y - X * beta;

    % Residual variance for each column (1 x K)
    sigma_squared = sum(resid_e.^2, 1) / (J - p);

    % Contrast variance term (scalar)
    contrast_var = c' * pinv(X' * X) * c;

    % Numerator of t-statistic (1 x K)
    num = c' * beta - d;

    % Denominator (1 x K)
    denom = sqrt(sigma_squared * contrast_var);

    % t-statistics
    t = num ./ denom;

    % Two-tailed p-values
    p_val = 2 * (1 - tcdf(abs(t), J - p));
end
