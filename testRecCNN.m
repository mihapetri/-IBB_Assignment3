fprintf('Recognition based on ID (type 1), ethnicity (type 2) or gender (type 3):\n');
str = input('', 's'); 
if str == '1'   
    fprintf('Recognition based on ID selected!\n');
    db = 'id';
    net = load('IDNet.mat').net;
    imTest = load('testSetI.mat').imTest;
    paths = imTest.Files;
    for i=1:length(imTest.Files)
        path = split(imTest.Files(i), 'id');
        path = char(append("id", path(2)));
        paths(i) = {char(path)};
    end
elseif str == '2'
    fprintf('Recognition based on ethnicity selected!\n')
    db = 'ethnicity';
    net = load('ethnicityNet.mat').net;
    imTest = load('testSetE.mat').imTest;
    paths = imTest.Files;
    for i=1:length(imTest.Files)
        path = split(imTest.Files(i), 'ethnicity');
        path = char(append("ethnicity", path(2)));
        paths(i) = {char(path)};
    end
elseif str == '3'
    fprintf('Recognition based on gender selected!\n');
    db = 'gender';
    net = load('genderNet.mat').net;
    imTest = load('testSetG.mat').imTest;
    paths = imTest.Files;
    for i=1:length(imTest.Files)
        path = split(imTest.Files(i), 'gender');
        path = char(append("gender", path(2)));
        paths(i) = {char(path)};
    end
end

paths = imageDatastore(paths);

YPredict = classify(net,paths);
YTest = imTest.Labels;

accuracy = sum(YPredict == YTest)/numel(YTest)

f = find(YPredict ~= YTest);
figure; 

for i = 1:6
    subplot(2,3,i);
    I = imread(cell2mat(paths.Files(f(i))));
    imshow(I);
    title([YPredict(f(i)), ' predicted, ', YTest(f(i)), ' actual']);
end
