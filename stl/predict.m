function ans = predict(theta, X)

  m = size(X, 2);

  [~, ans] = max(theta' * X, [], 1);
