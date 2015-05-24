function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
numLayers = numel(ei.layer_sizes) + 1;
hAct = cell(numLayers, 1);
gradStack = cell(numLayers-1, 1);
gradDelta = cell(numLayers, 1);
m = size(data, 2);

%% forward prop
%%% YOUR CODE HERE %%%

hAct{1} = data;
for d = 1:numHidden
  z{d+1} = stack{d}.W * hAct{d} + repmat(stack{d}.b, 1, m);
  hAct{d+1} = sigmoid(z{d+1});
end
d = d + 1;
z{d+1} = stack{d}.W * hAct{d} + repmat(stack{d}.b, 1, m);
expZ = exp(z{d+1});
hAct{d+1} = expZ ./ repmat(sum(expZ), size(expZ, 1), 1);

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  pred_prob = hAct{d+1};
  grad = [];
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
y_map = zeros(ei.output_dim, m);
y_map(sub2ind(size(y_map), labels', 1:m)) = 1;
cost = -1 / m * sum(y_map .* log(hAct{numLayers}));
% cost = -1 / m * sum(log(hAct{numLayers}(sub2ind(size(hAct{numLayers}), labels' , 1:m))));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
d = numLayers; % d is now the index of the output layer
gradDelta{d} = -(y_map - hAct{d});

for d = (numLayers - 1):1:2

  % Update the gradient stack
  gradStack{d}.W = (gradDelta{d+1} * hAct{d}') / m;
  gradStack{d}.b = sum(gradDelta{d+1}, 2) / m;

  % Update the discrete delta stack
  gradDelta{d} = (stack{d}.W' * gradDelta{d+1}) .* hAct{d} .* (1 - hAct{d});
end

d = 1;
gradStack{d}.W = (gradDelta{d+1} * hAct{d}') / m;
gradStack{d}.b = sum(gradDelta{d+1}, 2) / m;

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
for d = 1:(numLayers - 1)

  cost = cost + ei.lambda / 2 * sum(sum(stack{d}.W .^ 2));
  gradStack{d}.W = gradStack{d}.W + ei.lambda * stack{d}.W;
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end
