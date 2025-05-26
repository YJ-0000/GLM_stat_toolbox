%==========================================================================
% Function: t_test_GLM_permutation
%
% Description:
%   Performs a permutation-based t-test for a General Linear Model (GLM)
%   contrast. It uses a partitioning scheme to decompose the design matrix
%   and estimates the null distribution by permuting the reduced regressors.
%
% Syntax:
%   [t_obs, p_val, num_perm] = t_test_GLM_permutation(X, y, c)
%   [t_obs, p_val, num_perm] = t_test_GLM_permutation(X, y, c, d, num_perm, perm_method, partition_scheme)
%
% Inputs:
%   X               : [r x p] design matrix
%   y               : [r x 1] response vector
%   c               : [p x 1] contrast vector
%   d               : scalar, null hypothesis value (default = 0)
%   num_perm        : number of permutations (default = 10000)
%   perm_method     : string, permutation method ('smith') (default = 'smith')
%   partition_scheme: string, partitioning method ('Ridgway (2009)') (default)
%
% Outputs:
%   t_obs    : observed t-statistic for the contrast
%   p_val    : p-value from permutation test
%   num_perm : number of permutations used
%
% Example:
%   X = [1 0; 1 1; 1 2; 1 3];
%   y = [2; 2.5; 3; 4];
%   c = [0; 1];
%   [t_obs, p_val, num_perm] = t_test_GLM_permutation(X, y, c)
%
% Author:
%   Youngjo Song
%
% Date:
%   May 26, 2025
%==========================================================================

function [t_obs, p_val, num_perm] = t_test_GLM_permutation(X, y, c, d, num_perm, perm_method, partition_scheme)

    % Set default value for d (null hypothesis)
    if nargin < 4
        d = 0;
    end

    % Default number of permutations
    if nargin < 5
        num_perm = 10000;
    end

    % Default permutation method
    if nargin < 6
        perm_method = 'smith';
    end

    % Default partition scheme
    if nargin < 7
        partition_scheme = 'Ridgway (2009)';
    end

    % Partition the design matrix using Ridgway's method
    if strcmp(partition_scheme, 'Ridgway (2009)')
        % Extract the component of X corresponding to contrast c
        X1 = X * pinv(c)';
        
        % Project X onto the orthogonal complement of c
        X0 = X - X * c * pinv(c);
    else
        % Other partitioning methods not yet implemented
        error('Partition scheme not implemented');
    end

    r = size(X, 1); % Number of observations

    % Combine X1 and X0 to form the new design matrix
    X_new = [X1, X0];

    % Create a new contrast vector corresponding to X1
    c_new = zeros(size(X_new, 2), 1); 
    c_new(1) = 1;

    % The implementation assumes d == 0; raise error otherwise
    assert(d == 0, 'Non-zero null value d not supported.');

    % Compute observed t-statistic using the modified design
    [t_obs, ~] = t_test_GLM(X_new, y, c_new, d);

    % Permutation test using the 'smith' method
    if strcmp(perm_method, 'smith')
        % Compute residualized version of X1
        X1_resid = X1 - X0 * pinv(X0) * X1;

        % Initialize list to store permuted t-values
        t_perm_list = zeros(1, num_perm);

        for np = 1:num_perm
            % Permute rows of X1_resid and compute t-stat
            permuted_X1 = X1_resid(randperm(r), :);
            [t_perm, ~] = t_test_GLM([permuted_X1, X0], y, c_new, d);
            t_perm_list(np) = t_perm;
        end
    else
        % Other permutation methods not yet implemented
        error('Permutation method not implemented');
    end

    % Calculate p-value as proportion of permuted t-stats more extreme than observed
    p_val = mean(abs(t_perm_list) > abs(t_obs));
end
