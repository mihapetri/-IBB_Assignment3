fprintf('Recognition based on ID (type 1), ethnicity (type 2) or gender (type 3):\n');
str = input('', 's'); 
if str == '1'   
    fprintf('Recognition based on ID selected!\n');
    db = 'id';
    net = load('IDNet.mat').net;
    imTest = load('testSetI.mat').imTest;
elseif str == '2'
    fprintf('Recognition based on ethnicity selected!\n')
    db = 'ethnicity';
    net = load('ethnicityNet.mat').net;
    imTest = load('testSetE.mat').imTest;
elseif str == '3'
    fprintf('Recognition based on gender selected!\n');
    db = 'gender';
    net = load('genderNet.mat').net;
    imTest = load('testSetG.mat').imTest;
end

YPred = classify(net,imTest);
YTest = imTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)

ind = find(YPred ~= YTest);
figure; 

for ii = 1:6
    ind(ii)
    subplot(2,3,ii);
    imshow(imTest.readimage(ind(ii)));
    title([YPred(ind(ii)), ' predicted, ', YTest(ind(ii)), ' actual']);
end