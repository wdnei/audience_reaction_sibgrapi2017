function [ lbpFeatures ] = extrair_face_lbp_features( img1 )
%extrair_face_lbp_features 
% Extrair face e recuperar LBP features

if(size(img1,3)>1)
    img1 = rgb2gray(img1); % converter para gray
end

%% extrair face de imagem 
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MinSize',[100 100]); %Create a detector object

face_mask = step(faceDetector,img1); % detectar face

%iimg = insertObjectAnnotation(img1, 'rectangle', face_mask, 'Face'); %Annotate detected faces.

if(isempty(face_mask))
    lbpFeatures = [];
    return;
end

face = imcrop(img1,face_mask(1,:));

%% LBP 9 blocos
%face = imresize(face,[192 192],'nearest');
windowSize = fix(size(face,1)/3);

lbpFeatures = extractLBPFeatures(face,'CellSize',[windowSize windowSize],'Normalization','None');

%% split in 9 cells

%numNeighbors = 8;
%numBins = numNeighbors*(numNeighbors-1)+3;
%lbpCellHists = reshape(lbpFeatures,numBins,[]);


end

