function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
m = size(x, 2);
% Solve some linear algebra
% Note : the below computes covarience matrix
% with the assumption that the mean is zero
% Hence, normalization has to be done before hand (which is)

sigma = 1 / m * x * x';
[ U, S, V ] = svd(sigma);

% Calculate value of ZCA-Whittened image
Z = U * (diag(1 ./ sqrt(diag(S) + epsilon))) * U' * x;