fprintf('Recognition based on ID (type 1), ethnicity (type 2) or gender (type 3):\n');
str = input('', 's'); 
if str == '1'   
    fprintf('Recognition based on ID selected!\n');
    db = 'id';
    num_class = 100;
elseif str == '2'
    fprintf('Recognition based on ethnicity selected!\n')
    db = 'ethnicity';
    num_class = 6;
elseif str == '3'
    fprintf('Recognition based on gender selected!\n');
    db = 'gender';
    num_class = 2;
end

images = imageDatastore(db, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

TrainPart = 0.7;
ValPart = 0.15;
TestPart = 0.15;
[imTrain, imVal, imTest ] = splitEachLabel(images, ...
    TrainPart, ValPart, TestPart, 'randomize');

layers = [
    imageInputLayer([224 224 1])

    convolution2dLayer(3,10,'Padding','same')
    batchNormalizationLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,10,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(num_class)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MiniBatchSize', 64, ...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imVal, ...
    'LearnRateSchedule', 'piecewise', ...
    'ValidationFrequency',30, ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 100, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imTrain,layers,options);

if str == '1'   
    save IDNet.mat net   
elseif str == '2'
    save ethnicityNet.mat net
elseif str == '3'
    save genderNet.mat net
end
