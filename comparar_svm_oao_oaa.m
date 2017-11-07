%% Comparar o SVM Quadratico com abordagem Um-Contra-Um e Um-Contra-Todos


close all, clear all, clc, format compact

% industrial data
load lbp_olho_boca_3partes_rafd2_frontal
%load lbp_faces_rafd2
%load lbp_olho_boca_rafd2;

emotions_list_note_pt = {'neutro','raiva','desdenho','nojo','medo','alegria','tristeza','surpresa'};
emotions_list_note = {'neutral','angry','contemptuous','disgusted','fearful','happy','sad','surprised'};
emotions_list_class = [1,2,3,4,5,6,7,8];
map_note = containers.Map(emotions_list_class,emotions_list_note);
map_class = containers.Map(emotions_list_note,emotions_list_class);

N=10;

P = features;
T = target;
% Particionando os dados

p = cvpartition((T),'Holdout',0.30);

cp_svm_linear_oao = classperf(T);
cp_svm_linear_oaa = classperf(T);


%%
for k=1:N
    disp([num2str(k),'/',num2str(N)]);
    

    
    
    %% SVM multiclasse  quadratico UM-CONTRA-UM
    svm = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
    Mdlsvm = fitcecoc(P(p.training,:),T(p.training,:),'Learners',svm,'Coding', 'onevsone');
    labels = predict(Mdlsvm,P(p.test,:));
    errRate_svm_linear = sum(T(p.test) ~= labels)/p.TestSize;
    conMat_svm_linear = confusionmat(T(p.test),labels); % the confusion matrix

    classperf(cp_svm_linear_oao,labels,p.test);
    
    %% SVM multiclasse Quadrativo UM-CONTRA-TODOS
    
    svm = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
    Mdlsvm = fitcecoc(P(p.training,:),T(p.training,:),'Learners',svm,'Coding', 'onevsall');
    labels = predict(Mdlsvm,P(p.test,:));
    errRate_svm_linear = sum(T(p.test) ~= labels)/p.TestSize;
    conMat_svm_linear = confusionmat(T(p.test),labels); % the confusion matrix

    classperf(cp_svm_linear_oaa,labels,p.test);
    
    
%%
    p = cvpartition(T,'Holdout',0.30);
end

%% plotar resultado

%% UM-CONTRA-UM
cp=cp_svm_linear_oao;
desenha_matriz_confusao( cp ,emotions_list_note_pt )
title(['Matriz Confusão SVM Quadrático Um-Contra-Um (iterações N=',num2str(N),', Tx. Acerto ',num2str(cp.CorrectRate*100),'%)']);

%% UM-CONTRA-TODOS
cp=cp_svm_linear_oaa;
desenha_matriz_confusao( cp ,emotions_list_note_pt )
title(['Matriz Confusão SVM Quadrático Um-Contra-Todos (iterações N=',num2str(N),', Tx. Acerto ',num2str(cp.CorrectRate*100),'%)']);
