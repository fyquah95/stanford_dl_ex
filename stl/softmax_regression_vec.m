function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  theta = [ theta zeros(size(theta), 1) ];
  % Completed parameter initialization

  mat = theta' * X;
  h = exp(mat) ./ repmat(sum(exp(mat)), num_classes, 1);
  f = -sum(log(h(sub2ind(size(h), y, 1:m))));

  y_filter = zeros(num_classes, m);
  y_filter(sub2ind(size(y_filter), y, 1:m)) = 1;

  g = -1 * X * (y_filter - h)';
  g = g(:, 1:num_classes - 1);
  g=g(:); % make gradient a vector for minFunc
