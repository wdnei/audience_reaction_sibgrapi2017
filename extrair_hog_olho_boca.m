function [ hogFeatures,nova_face ] = extrair_hog_olho_boca( img1 )
%EXTRAIR_hog_OLHO_BOCA Summary of this function goes here
%   Detailed explanation goes here


if(size(img1,3)>1)
    img1 = rgb2gray(img1); % converter para gray
end

faceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MinSize',[100 100]); %Create a detector object
face_mask = step(faceDetector,img1); % detectar face
%iimg = insertObjectAnnotation(img1, 'rectangle', face_mask, 'Face'); %Annotate detected faces.

if(isempty(face_mask))
    hogFeatures = [];
    return;
end


face = imcrop(img1,face_mask(1,:));

eyeDetector = vision.CascadeObjectDetector('EyePairBig'); %Create a detector object

eye_mask = step(eyeDetector,face); % detectar olho

% faz outra tentativa com a imagem inteira
% e mostra o resultado
if(isempty(eye_mask))
    eye_mask = step(eyeDetector,img1); % detectar olho
    rosto = insertObjectAnnotation(img1, 'rectangle', face_mask(1,:), 'Face'); %Annotate detected faces.
    olho = insertObjectAnnotation(rosto, 'rectangle', eye_mask, 'Olho'); %Annotate detected faces.
    h=figure;
    imshow(olho);
    choice = menu('Outra tentativa de encontrar o olho foi realizada, esta correta?','Sim','Não');
    if choice==2 || choice==0
        close(h);
       hogFeatures = [];
        return;
    end
    close(h);
end


if(isempty(eye_mask))
    hogFeatures = [];
    return;
end

% aumentar mascara do olho
subtrairAltura = fix((40*size(face,1))/356);
adicionarAltura = fix((60*size(face,1))/356);
marcaraOlho =[eye_mask(1) eye_mask(2)-subtrairAltura eye_mask(3) eye_mask(4)+adicionarAltura];

%adicionar 5% nas laterais

perc= fix(eye_mask(3)*0.05);
marcaraOlho(3) = marcaraOlho(3) + perc*2;
marcaraOlho(1) = marcaraOlho(1) - perc;

%face = insertObjectAnnotation(face, 'rectangle', marcaraOlho, 'Olho'); %Annotate detected faces.
%figure;

% mascara da boca
X = [eye_mask(1:2);1 size(face,1)];
d = pdist(X,'euclidean');
mascaraBoca = [eye_mask(1) fix((eye_mask(2)+eye_mask(4)+d)/2) eye_mask(3) fix(d/2)];

%remover 15% das laterais
perc= fix(eye_mask(3)*0.15);
mascaraBoca(3) = mascaraBoca(3) - perc*2;
mascaraBoca(1) = mascaraBoca(1) + perc;


if(isempty(mascaraBoca))
    hogFeatures = [];
    return;
end
%face = insertObjectAnnotation(face, 'rectangle', mascaraBoca, 'Boca'); %Annotate detected faces.
%imshow(face);


%% hog - cada imagem foi dividida em 9 celulas com uma celular por bloco, assim temos 9celulasX9blocos=81 regioes  81regioesX9Bins=729atributos

croppedImage = imcrop(face,marcaraOlho(1,:));
windowSize = fix(size(croppedImage,1)/3);
[hogFeaturesOlho,hogVisualization ]= extractHOGFeatures(croppedImage,'CellSize',[fix(size(croppedImage,1)/9) fix(size(croppedImage,2)/9)],'BlockSize',[9 9]);



croppedImage = imcrop(face,mascaraBoca(1,:));
[hogFeaturesBoca,hogVisualization ]= extractHOGFeatures(croppedImage,'CellSize',[fix(size(croppedImage,1)/9) fix(size(croppedImage,2)/9)],'BlockSize',[9 9]);


hogFeatures = [hogFeaturesOlho hogFeaturesBoca];

nova_face=insertObjectAnnotation(face, 'rectangle', marcaraOlho, 'Olho'); %Annotate detected faces.
nova_face = insertObjectAnnotation(nova_face, 'rectangle', mascaraBoca, 'Boca'); %Annotate detected faces.



end

