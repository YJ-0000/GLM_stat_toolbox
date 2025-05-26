%==========================================================================
% Function: F_test_GLM
%
% Description:
%   Performs an F-test for a general linear model (GLM) to assess whether 
%   a set of linear constraints (represented by matrix C) significantly 
%   explains variance in the dependent variable y.
%
% Syntax:
%   [F, p_val] = F_test_GLM(X, y, C)
%
% Inputs:
%   X : [J x p] design matrix (independent variables)
%   y : [J x 1] response vector (observed data)
%   C : [p x q] contrast matrix representing linear hypotheses to test
%       (e.g., C' * beta = 0)
%
% Outputs:
%   F     : F-statistic for the hypothesis test
%   p_val : p-value associated with the F-statistic (right-tailed test)
%
% Author:
%   Youongjo Song
%
% Date:
%   May 26, 2025
%==========================================================================

function [F, p_val] = F_test_GLM(X, y, C)
    % Number of observations
    J = size(X, 1);

    % Rank of the full design matrix
    p = rank(X);

    % Construct a projection matrix to remove the contrast space (C0)
    C0 = eye(size(X, 2)) - C * pinv(C);

    % Reduced design matrix under the null hypothesis (removing contrast effect)
    X0 = X * C0;

    % Rank of the reduced design matrix
    p2 = rank(X0);

    % Degrees of freedom for the effect being tested
    p1 = p - p2;

    % Residual forming matrix for the reduced model
    R0 = eye(J) - X0 * pinv(X0);

    % Residual forming matrix for the full model
    R = eye(J) - X * pinv(X);

    % Difference in projection matrices (represents variance explained by contrast)
    M = R0 - R;

    % F-statistic calculation
    F = ((J - p) / p1) * (y' * M * y) / (y' * R * y);

    % p-value from the F-distribution
    p_val = 1 - fcdf(F, p1, J - p);
end
