%% EXTRAIR FACES E SALVAR LBP EM MATRIX
emotions_list_note = {'neutral','angry','contemptuous','disgusted','fearful','happy','sad','surprised'};
emotions_list_class = [1,2,3,4,5,6,7,8];
map_note = containers.Map(emotions_list_class,emotions_list_note);
map_class = containers.Map(emotions_list_note,emotions_list_class);
%%

dr=dir('../datasets/rafd/*frontal.jpg');

features = zeros(length(dr),59*9);
target = zeros(length(dr),1);

if length(dr)>0
     nm=[];
    for fc = 1:length(dr)
        fullFile = fullfile(dr(fc).folder,dr(fc).name);
        fln = dr(fc).name;
        lfln = length(fln);        
        ns = fln(1:lfln-4); % remover extensao                       
        name_parts = strsplit(ns,'_');
        
        emotion_note = name_parts{5};
        
        %ler imagem
        img1 = imread(fullFile);
        %extrair feature
        lbp_features = extrair_face_lbp_features(img1);
        if(~isempty(lbp_features))
            features(fc,:) = lbp_features;
            target(fc) = map_class(emotion_note);
        else
            disp(['face não encontrada:',fullFile])
        end
        
        disp(fc);
        
    end
end

save('lbp_faces_rafd2.mat','features','target');
disp('Terminou');


%% extrair face de imagem
%img1 = imread('Rafd090_01_Caucasian_female_neutral_frontal.jpg'); %Read input image
%img1 = imread('Rafd090_07_Caucasian_male_neutral_frontal.jpg'); %Read input image
img1 = imread('Rafd090_41_Kid_female_neutral_frontal.jpg'); %Read input image
img1 = imread('Rafd090_11_Kid_female_disgusted_right.jpg'); %Read input image

faceDetector = vision.CascadeObjectDetector('FrontalFaceCART'); %Create a detector object
face_mask = step(faceDetector,img1); % detectar face
iimg = insertObjectAnnotation(img1, 'rectangle', face_mask, 'Face'); %Annotate detected faces.
%imshow(iimg);

face = imcrop(img1,face_mask(1,:));

eyeDetector = vision.CascadeObjectDetector('EyePairBig'); %Create a detector object

eye_mask = step(eyeDetector,face); % detectar olho

% aumentar mascara do olho
subtrairAltura = fix((40*size(face,1))/356);
adicionarAltura = fix((60*size(face,1))/356);
marcaraOlho = [fix(eye_mask(3)/2)-fix(eye_mask(3)/4) eye_mask(2)-subtrairAltura fix(eye_mask(3)/2)+fix(eye_mask(3)/4)*2 eye_mask(4)+adicionarAltura];




face = insertObjectAnnotation(face, 'rectangle', marcaraOlho, 'Olho'); %Annotate detected faces.
figure;




% mascara da boca
X = [eye_mask(1:2);1 size(face,1)];
d = pdist(X,'euclidean');
mascaraBoca = [fix(eye_mask(3)/2)-fix(eye_mask(3)/4) fix((eye_mask(2)+eye_mask(4)+d)/2) fix(eye_mask(3)/2)+fix(eye_mask(3)/4)*2 fix(d/2)];
face = insertObjectAnnotation(face, 'rectangle', mascaraBoca, 'Boca'); %Annotate detected faces.
imshow(face);


%% LBP 

croppedImage = imcrop(face,marcaraOlho(1,:));
croppedImage = rgb2gray(croppedImage);
lbpFeaturesOlho = extractLBPFeatures(croppedImage,'Normalization','None');


croppedImage = imcrop(face,mascaraBoca(1,:));
croppedImage = rgb2gray(croppedImage);
lbpFeaturesBoca = extractLBPFeatures(croppedImage,'Normalization','None');

%%

if(size(img1,3)>1)
    img1 = rgb2gray(img1); % converter para gray
end

face_mask = step(faceDetector,img1); % detectar face

iimg = insertObjectAnnotation(img1, 'rectangle', face_mask, 'Face'); %Annotate detected faces.

face = imcrop(img1,face_mask(1,:));


%% split in 9 cells

numNeighbors = 8;
numBins = numNeighbors*(numNeighbors-1)+3;
lbpCellHists = reshape(lbpFeatures,numBins,[]);

%%

hh=LBP(face,1);
imshow(uint8(hh));

%% optical flow Lucas-Kanade

opticFlow = opticalFlowLK('NoiseThreshold',0.009);
for k=1:size(img1,3)
    frame = img1(:,:,k);
    flow = estimateFlow(opticFlow,frame);

    imshow(frame);
    hold on;
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',2);
    hold off;
    d = waitforbuttonpress;

end

%%
img1 = imread('Rafd090_01_Caucasian_female_neutral_frontal.jpg'); %Read input image
[ lbpFeatures,nova_face ] = extrair_lbp_olho_boca( img1 );
imshow(nova_face);


%%

img1 = imread('Rafd090_01_Caucasian_female_neutral_frontal.jpg'); %Read input image
[ hogFeatures,nova_face ] = extrair_hog_olho_boca( img1 );
%imshow(nova_face);
