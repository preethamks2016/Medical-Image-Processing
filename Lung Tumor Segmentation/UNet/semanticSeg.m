% Triangle segmentation usig semantic segmentation    
% dataSetDur = fullfile('/usr2/prouserdata/preetham/');
dataSetDir = fullfile('./');
imageDir = fullfile(dataSetDir, './train/train_X');
labelDir = fullfile(dataSetDir,'./train/train_Y');

imds = imageDatastore(imageDir);
numberTrainImages = numel(imds.Files);

figure
trainImage = readimage(imds, 10);
imshow(trainImage)

classNames = ["tumor", "nontumor"];
labelIDs   = [255 0];
        
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

figure
trainY = readimage(pxds, 10);
trainYactual = uint8(trainY);
imagesc(trainYactual)

valImagDir = fullfile(dataSetDir, './cv/cv_X');
valLabelDir = fullfile(dataSetDir, './cv/cv_Y');
valImds = imageDatastore(valImagDir);
valPxds = pixelLabelDatastore(valLabelDir, classNames, labelIDs);
valPximds = pixelLabelImageDatastore(valImds, valPxds);

% -------------- Weighting the classes by term frequency matrix ----------------------------
% Class weighting by inverse class frequency
tb1 = countEachLabel(pxds);
totalNumberOfPixels = sum(tb1.PixelCount);
frequency = tb1.PixelCount/totalNumberOfPixels;
classWeight = 1./frequency;

%frequency = tb1.PixelCount/sum(tb1.PixelCount);

bar(1:numel(classNames), frequency)
xticks(1:numel(classNames))
xticklabels(tb1.Name)
xtickangle(45)
ylabel('Frequency')

%imageFreq = tb1.PixelCount ./ tb1.ImagePixelCount;
%classWeight = median(imageFreq) ./ imageFreq;
% -------------------------------------------------------------------------------------------

% classWeight = [2.5800 0.5611]  % vocalSpectNet2   % Changed class weights
% classWeight = [1.5 0.5611]

imageSize = [263 384];
numClasses = 2;
lgraph = segnetLayers(imageSize, numClasses, 6);

pximds = pixelLabelImageDatastore(imds, pxds);

% options = trainingOptions('adam','InitialLearnRate',1e-3, ...
%       'MaxEpochs',10, 'MiniBatchSize', 4, 'VerboseFrequency',10);
  
options = trainingOptions('rmsprop', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 30, ...
    'L2Regularization', 0.001, ...
    'MaxEpochs', 50, ...  
    'ValidationData', valPximds, ...
    'ValidationFrequency', 300,...
    'MiniBatchSize', 2, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 2, ...
    'Plots', 'training-progress',...
    'ExecutionEnvironment','gpu');

%layers(end) = pixelClassificationLayer('ClassNames', tbl.Name, 'ClassWeights', classWeight);

pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tb1.Name,'ClassWeights',classWeight);

lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax', 'labels');

% To plot the newtwork
figure
plot(lgraph)
  
net = trainNetwork(pximds, lgraph, options);

%save lungTumor

load lungTumor.mat
% --------------------- Evaluation part ---------------------------------------------

% testImageDir = fullfile(dataSetDir, 'test_x');
% testPixelImageDir = fullfile(dataSetDir, 'test_y');
testImageDir = fullfile(dataSetDir, './test/test_X');
testPixelImageDir = fullfile(dataSetDir, './test/test_Y');
imdsTest = imageDatastore(testImageDir);
imdsTestPixl = pixelLabelDatastore(testPixelImageDir, classNames, labelIDs);
%pxdsResults = semanticseg(imdsTest, net, 'WriteLocation', './vocal_harmonics/predictions','Verbose', fcvtrue);

% Running semantic segmentation on all the test images
writeLocation = './predictions';
pxdsResults = semanticseg(imdsTest, net, 'WriteLocation', writeLocation, 'MiniBatchSize', 4);
metrics = evaluateSemanticSegmentation(pxdsResults, imdsTestPixl);
save vocalFoldsMetrics
%load vocalFoldsMetrics.mat
% ------------------------------------------------------------------------------------------------

% ------------------- Plotting results -----------------------------------------------------------
% Normalized confusion matrix
normConfMatData = metrics.NormalizedConfusionMatrix.Variables;
figure
h = heatmap(classNames, classNames, 100 * normConfMatData);
h.XLabel = 'Predicted Classes';
h.YLabel = 'True classes';
h.Title = 'Normalized Confusion Matrix (%)';

% IoU histogram for each image
imageIou = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIou)
title('Image Mean IoU')

% Image with minimum IoU
[minIou, worstImageIndex] = min(imageIou);
minIou = minIou(1);
worstImageIndex = worstImageIndex(1);

worstTestImage = readimage(imdsTest, worstImageIndex);
worstTrueLabels = readimage(imdsTestPixl, worstImageIndex);
worstPredictedLabels = readimage(pxdsResults, worstImageIndex);

% Categorical labels to numerical values 
worstTrueLabelsImage = im2uint8(worstTrueLabels == classNames(1));
worstPredictedLabelsImage = im2uint8(worstPredictedLabels == classNames(1));

% Montage display of the ground truth and the predicted results 
% works if the image dimensions are all same. We have input spectrogram with m x n x 3 matrix
% worstMontage = cat(3, worstTestImage, worstTrueLabelsImage, worstPredictedLabelsImage);
% worstMontage = imresize(worstMontage, 4, "nearest");
% figure
% montage(worstMontage, 'Size', [1, 3]);
% title(['Test image vs. Truth vs. Prediction vs. IoU =' num2str(minIou)])
figure
imshow(worstTestImage)
figure
imshow(worstTrueLabelsImage)
figure
imshow(worstPredictedLabelsImage)

% Image with maximum IoU
[minIou, worstImageIndex] = max(imageIou);
minIou = minIou(1);
worstImageIndex = worstImageIndex(1);

worstTestImage = readimage(imdsTest, worstImageIndex);
worstTrueLabels = readimage(imdsTestPixl, worstImageIndex);
worstPredictedLabels = readimage(pxdsResults, worstImageIndex);

% Categorical labels to numerical values 
worstTrueLabelsImage = im2uint8(worstTrueLabels == classNames(1));
worstPredictedLabelsImage = im2uint8(worstPredictedLabels == classNames(1));

% Montage display of the ground truth and the predicted results 
% works if the image dimensions are all same. We have input spectrogram with m x n x 3 matrix
% worstMontage = cat(3, worstTestImage, worstTrueLabelsImage, worstPredictedLabelsImage);
% worstMontage = imresize(worstMontage, 4, "nearest");
% figure
% montage(worstMontage, 'Size', [1, 3]);
% title(['Test image vs. Truth vs. Prediction vs. IoU =' num2str(minIou)])
figure
subplot(1, 3, 1)
imshow(worstTestImage)
subplot(1, 3, 2)
imshow(worstTrueLabelsImage)
subplot(1, 3, 3)
imshow(worstPredictedLabelsImage)
% -------------------------------------------------------------------------------------------------
% TODO: Problem in computing Jaccard matrix

for i = 1:length(imdsTest.Files)
    
    testImage = readimage(imdsTest, i);
    %imwrite(imresize(testImage, [150, 150]), strcat('./results/', num2str(i), '.jpg'))
    
    %imshow(testImage)
    testGroundTruth = readimage(imdsTestPixl, i);
    [C, scores] = semanticseg(testImage, net);

%     iou = jaccard(C, testGroundTruth);
%     table(classNames', iou)
     
    testGroundTruth = uint8(testGroundTruth);
    testGroundTruth = 2 - testGroundTruth;
    testGroundTruth(testGroundTruth == 2) = 255;
    %imshow(testGroundTruth)
    
    actual = 2 - uint8(C);
    segmentedFolds = actual .* testImage;
    
    %imshow(segmentedFolds)
    %imwrite(imresize(segmentedFolds, [150, 150]), strcat('./results/', num2str(i), '_seg', '.jpg'))
    
    %testGroundTruth = 255;
    actual(actual == 2) = 255; 
    
    %B = labeloverlay(testImage, C, 'Transparency', 0.99);
    
    %imwrite(imresize(segmentedFolds, [150, 150]), strcat('./results/', num2str(i), '_seg', '.jpg'))

%     figure()
%     imshow(B)
    
%     subplot(1, 3, 1)
%     imshow(testImage)
%     subplot(1, 3, 2)
%     imshow(testGroundTruth)
%     subplot(1, 3, 3)
%     imshow(actual)
    figure()
    imshowpair(testImage, testGroundTruth, 'montage');
    title(['Test Image            ', '            Ground Truth']);
    
    figure()
    imshowpair(testImage, actual, 'montage');
    title(['Test Image            ', '            Predicted']);
% 
%     figure()
%     imshowpair(testGroundTruth, actual, 'montage')
%     title(['Ground truth            ', '            Predicted']);
    
    pause

%     BW = C == 'harmonic'; 
%     imshow(BW)
%     imagesc(scores)
%     axis square 
%     colorbar
%     pause(2)
    
end

% -----------------------------------------------------------------------------------
% Testing network properties
% -----------------------------------------------------------------------------------

net.Layers(1:5)
net.Layers(1)
net.Layers(2)
act = activations(net, testImage, 'encoder6_relu_1', 'OutputAs', 'channels');
size(act)
min(act(:))
max(act(:))

act = reshape(act,size(act,1),size(act,2),1,size(act,3));
act_scaled = mat2gray(act);
montage(act_scaled)

tmp = act_scaled(:);
tmp = imadjust(tmp,stretchlim(tmp));
act_stretched = reshape(tmp,size(act_scaled));
montage(act_stretched)
title('Activations from the encoder1_conv1 layer','Interpreter','none')

figure
subplot(1,2,1)
imshow(act_stretched(:,:,:,33))
title('Channel 33')
subplot(1,2,2)
imshow(act_stretched(:,:,:,34))
title('Channel 34')

for i = 1:12
%i = 102;
    
    testImage = readimage(imdsTest, i);
    figure 
    imshow(testImage)
    
    act2 = activations(net, testImage,'decoder6_conv1');
    act2 = reshape(act2, size(act2, 1), size(act2, 2), 1, size(act2, 3));
    act2_scaled = mat2gray(act2);
    tmp = act2_scaled(:);
    lim = stretchlim(tmp);
    lim(1) = 0;
    tmp = imadjust(tmp, lim);
    act2_stretched = reshape(tmp, size(act2_scaled));

    figure
    montage(act2_stretched)
    title('Activations from the encoder1_relu_1 layer','Interpreter','none')

    figure
    %subplot(1,2,1)
    imshow(act2_stretched(:, :, :, 8))
    %title('Channel 33')
    %subplot(1,2,2)
    %imshow(act2_stretched(:, :, :, 8))
    %title('Channel 34')
    pause
end










