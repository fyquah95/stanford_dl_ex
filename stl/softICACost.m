%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);
m = size(x, 2);
lambda = params.lambda;
epsilon = params.epsilon;
% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

% % project weights to norm ball (prevents degenerate bases)
% Wold = W;
% W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%

Y = W * x;

cost = lambda * sum(sum(sqrt(Y .^ 2 + epsilon))) + 1 / (2*m) * sum(sum((W' * Y - x).^ 2));
Wgrad = lambda * (Y ./ sqrt(Y .^ 2 + epsilon) * x') + 1 / m * (Y*Y'*W + W*W'*Y*x' - 2*Y*x');

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
