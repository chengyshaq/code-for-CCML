% Implementation of Causal Inference on Discrete Data via Estimating Distance Correlations
% V contains the casual direction, 0 indicates shows correlation not causation,
% 1 indicates row to col, -1 indicates col to row,
% Only the value of the upper triangular matrix is valid
function [V, E] = causalInference(Y)
trainLabels = Y';
[labelSize, sampleSize] = size(trainLabels);
V = zeros(labelSize, labelSize);
E = zeros(labelSize, labelSize);

% Generate observation pairs
count = 1;
cellSize = (labelSize * labelSize - labelSize) / 2;
observations = cell(1, cellSize);
tempObservation = zeros(2, sampleSize);
for i = 1 : labelSize
    n = i + 1;
    tempObservation(1, :) = trainLabels(i, :);
    for j = n : labelSize
        tempObservation(2, : ) = trainLabels(j, :);
        observations{count} = tempObservation;
        count = count + 1;
    end
end

directions = zeros;
for i = 1 : cellSize
    % Generate possibility distribution pair
    observation = observations{1, i};
    observation1 = observation(1, :);
    observation2 = observation(2, :);
    distributionPair1 = GetDistributionPair(observation1, observation2);
    distributionPair2 = GetDistributionPair(observation2, observation1);
    
    % Generate matrix for causal direction inference
    matrixA1 = ConstructMatrix(distributionPair1(:, 1));
    matrixB1 = ConstructMatrix(distributionPair1(:, 2 : 3));
    matrixA2 = ConstructMatrix(distributionPair2(:, 1));
    matrixB2 = ConstructMatrix(distributionPair2(:, 2 : 3));
    
    % Infer causal direction
    directions(i) = InferDirection(matrixA1, matrixB1, matrixA2, matrixB2);
end

count = 1;
for m = 1 : labelSize
    f = m + 1;
    for n = f : labelSize
        if (directions(count) == 1)
            V(m, n) = 1;
        elseif (directions(count) == -1)
            V(n, m) = 1;
        end

        count = count + 1;
    end
end

% view(biograph(V));
end

function distributionPair = GetDistributionPair(observation1, observation2)
% a = a0, a1
obA = observation1;
obB = observation2;
table = tabulate(obA);
percentage = GetPercentage(table);
a = percentage / 100;

% b = b0_a0, b1_a0
index = find(obA == -1);
b_a0 = obB(:, index);
table = tabulate(b_a0);
percentage = GetPercentage(table);
b = percentage / 100;

% c = b0_a1, b1_a1;
index = find(obA == 1);
b_a1 = obB(:, index);
table = tabulate(b_a1);
percentage = GetPercentage(table);
c = percentage / 100;

% distributionPair = [[a, a]; [b, c]]';
distributionPair = [[a(1), b]; [a(2), c]];
end

function percentage = GetPercentage(table)
count = size(table, 1);
percentage = zeros(1, 2);

for i = 1 : count
    if (table(i, 1) == -1)
        percentage(1, 1) = table(i, 3);
    elseif (table(i, 1) == 1)
        percentage(1, 2) = table(i, 3);
    end
end
end

function matrixN = ConstructMatrix(data)
dataSize = size(data, 1);
matrix = zeros(dataSize, dataSize);
temp = zeros(dataSize, dataSize);
for i = 1 : dataSize
    ai = data(i, :);
    for j = 1 : dataSize
        aj = data(j, :);
        temp(i, j) = norm(ai - aj, 2);
    end
end

a = abs(temp);
ai_ = sum(a) / dataSize;
a_j = sum(a, 2) / dataSize;
a__ = sum(sum(a)) / (dataSize * dataSize);
for i = 1 : dataSize
    for j = 1 : dataSize
        matrix(i, j) = a(i, j) - ai_(i) - a_j(j) + a__;
    end
end
matrixN = abs(matrix);
end

function direction = InferDirection(a1, b1, a2, b2, threshold)
n = 2;
cab1 = sqrt(sum(sum(a1 * b1))) / n;
caa1 = sqrt(sum(sum(a1 * a1))) / n;
cbb1 = sqrt(sum(sum(b1 * b1))) / n;
d1 = cab1 / (sqrt(caa1 * cbb1));

cab2 = sqrt(sum(sum(a2 * b2))) / n;
caa2 = sqrt(sum(sum(a2 * a2))) / n;
cbb2 = sqrt(sum(sum(b2 * b2))) / n;
d2 = cab2 / (sqrt(caa2 * cbb2));

direction = 0;
% decision = d1 - d2;
if (d1 > d2)
    direction = -1;
elseif (d2 > d1)
    direction = 1;
end
end