pkg load statistics;

filterDim = 20;
imageDim = 999;
numFilters = 2;
numImages = 3;
convolvedDim = (imageDim - filterDim + 1);
poolDim = 20;

filters = [];
images = [];
b = zeros(numFilters, 1);

for idx = 1:numImages
  images(:, :, idx) = abs(rand(imageDim) * 200);
end

for idx = 1:numFilters
  filters(:, :, idx) = normrnd(0, 0.0001, filterDim, filterDim);
end

convolutedCollection = cnnConvolve(filterDim, numFilters, images, filters, b);
sum_difference = 0;

printf('Checking convolution ...\n');
for imageNum = 1:numImages
  for filterNum = 1:numFilters
    convoluted = convolutedCollection(:, :, filterNum, imageNum);

    im = images(:, :, imageNum);
    f = rot90(filters(:, :, filterNum), 2);

    err = abs(convoluted - sigmoid(b(filterNum) + conv2(im, f, 'valid')));
    err = sum(sum(err));

  end
end

average_difference = sum_difference / (numImages * numFilters * imageDim * imageDim);

printf('The average error in the calculation : %14f\n', average_difference);
assert(abs(average_difference) < 1e-10);


printf('Checking pooling now\n');

pooledFeatures = cnnPool(poolDim, convolutedCollection);
sum_difference = 0;

for imageNum = 1:numImages
  for filterNum = 1:numFilters

    for checks = 1:5 
      % Check only 5 pools per image per filter
      a = randsample(1:(convolvedDim / poolDim - 1), 1);
      b = randsample(1:(convolvedDim / poolDim - 1), 1);

      convoluted = convolutedCollection( 
        (1 + (a-1) * poolDim):(a * poolDim), ...
        (1 + (b-1) * poolDim):(b * poolDim), ...
        filterNum, imageNum 
      );

      res = sum(sum(convoluted)) / (poolDim * poolDim);
      
      sum_difference = sum_difference + abs(pooledFeatures(a, b, filterNum, imageNum) - res);
      assert(abs(pooledFeatures(a, b, filterNum, imageNum) - res) < 1e-10);
    end

  end
end

printf('Compelted pooling check!\n');
printf('Average difference is : %14f\n', sum_difference / (numImages * numFilters * 5));