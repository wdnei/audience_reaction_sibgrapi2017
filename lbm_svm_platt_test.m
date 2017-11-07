%% Resolução da questão 2 da letra a) até c)
% Definindo o codigo de saida das classes
close all, clear all, clc, format compact

% industrial data
%load lbp_olho_boca_3partes_rafd2_frontal;
load hog_olho_boca_3partes_rafd2_frontal;
emotions_list_note_pt = {'neutro','raiva','desdenho','nojo','medo','alegria','tristeza','surpresa'};
emotions_list_note = {'neutral','angry','contemptuous','disgusted','fearful','happy','sad','surprised'};
emotions_list_class = [1,2,3,4,5,6,7,8];
map_note = containers.Map(emotions_list_class,emotions_list_note_pt);
map_class = containers.Map(emotions_list_note,emotions_list_class);

P = features;
T = target;
% Particionando os dados

p = cvpartition((T),'Holdout',0.30);

cp_svm_linear = classperf(T);

%% one vs all SVM QUADRATICO
    template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    P(p.training,:), ...
    T(p.training,:), ...
    'FitPosterior',1, ...
    'Learners', template, ...
    'Coding', 'onevsall');


    [labels,NegLoss,PBScore,Posterior] = predict(classificationSVM,P(p.test,:));
    errRate_svm_poly = sum(T(p.test) ~= labels)/p.TestSize;
    conMat_svm_poly = confusionmat(T(p.test),labels); % the confusion matrix

    classperf(cp_svm_linear,labels,p.test);

%% mostrar resultados
desenha_matriz_confusao(cp_svm_linear,emotions_list_note);

%%  Treinar com toda a base e testar imagens externas

template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    P, ...
    T, ...
    'FitPosterior',1, ...
    'Learners', template, ...
    'Coding', 'onevsall');

%%
img1 = imread('S026_002_01384609.png');
%img1 = imread('Rafd090_68_Moroccan_male_sad_right.jpg');
%img1 = imread('Rafd090_68_Moroccan_male_sad_right.jpg');
%img1 = imread('S014_004_02404104.png');
%img1 = imread('S106_002_00460429.png');
%img1 = imread('Rafd090_68_Moroccan_male_sad_right.jpg');
img1 = imread('S108_002_01103905.png');

[lbpFeatures,nova_face]=extrair_hog_olho_boca(img1);
[labels,NegLoss,PBScore,Posterior] = predict(classificationSVM,[lbpFeatures]);

figure;
subplot(211);
imshow(nova_face);
title(['Predicão:',(map_note(labels)),' ',num2str(Posterior(labels)*100),'%']);
subplot(212);
h=bar(Posterior);
set(gca,'XTick',1:length(emotions_list_note_pt),...                         %# Change the axes tick marks
        'XTickLabel',emotions_list_note_pt);




