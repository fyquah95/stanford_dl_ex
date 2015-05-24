function average = run_check()
  addpath ../common;
  addpath(genpath('../common/minFunc_2012/minFunc'));

  ei = [];
  % dimension of input features
  ei.input_dim = 784;
  % number of output classes
  ei.output_dim = 10;
  % sizes of all hidden layers and the output layer
  ei.layer_sizes = [10, ei.output_dim];
  % scaling parameter for l2 weight regularization penalty
  ei.lambda = 0;
  % which type of activation function to use in hidden layers
  % feel free to implement support for only the logistic sigmoid function
  ei.activation_fun = 'logistic';

  %% setup random initial weights
  stack = initialize_weights(ei);
  [data_train, labels_train, data_test, labels_test] = ...
    load_preprocess_mnist();

  stack = initialize_weights(ei);
  params = stack2params(stack);

  average = grad_check(@supervised_dnn_cost, params, 10, ...
    ei, data_train, labels_train);
end
