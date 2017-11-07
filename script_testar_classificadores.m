%% usando SVM com a resolução de Platt, onde tempos a saída com porcetagens
close all, clear all, clc, format compact

% industrial data
%load lbp_olho_boca_3partes_rafd2_frontal
load hog_olho_boca_3partes_rafd2_frontal


emotions_list_note_pt = {'neutro','raiva','desdenho','nojo','medo','alegria','tristeza','surpresa'};
emotions_list_note = {'neutral','angry','contemptuous','disgusted','fearful','happy','sad','surprised'};
emotions_list_class = [1,2,3,4,5,6,7,8];
map_note = containers.Map(emotions_list_class,emotions_list_note);
map_class = containers.Map(emotions_list_note,emotions_list_class);

N=100;

P = features;
T = target;
% Particionando os dados

p = cvpartition((T),'Holdout',0.30);

cp_svm_linear = classperf(T);
cp_svm_rbf = classperf(T);
cp_svm_poly = classperf(T);
cp_rna = classperf(T);
cp_naive = classperf(T);


%imgT =imread('S052_001_00115103.png');



%%
for k=1:N
    disp([num2str(k),'/',num2str(N)]);
    
    %% one vs one SVM QUADRATICO
    template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    P(p.training,:), ...
    T(p.training,:), ...
    'Learners', template, ...
    'Coding', 'onevsall');


    [labels,score] = predict(classificationSVM,P(p.test,:));
    errRate_svm_poly = sum(T(p.test) ~= labels)/p.TestSize;
    conMat_svm_poly = confusionmat(T(p.test),labels); % the confusion matrix

    classperf(cp_svm_poly,labels,p.test);

    
    
    %% SVM multiclasse linear
    svm = templateSVM(...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
    Mdlsvm = fitcecoc(P(p.training,:),T(p.training,:),'Learners',svm,'Coding', 'onevsall');
    labels = predict(Mdlsvm,P(p.test,:));
    errRate_svm_linear = sum(T(p.test) ~= labels)/p.TestSize;
    conMat_svm_linear = confusionmat(T(p.test),labels); % the confusion matrix

    classperf(cp_svm_linear,labels,p.test);
    

    %% SVM multiclasse RBF
    svm = templateSVM('SaveSupportVectors',1,'KernelFunction','rbf');
    Mdlsvm = fitcecoc(P(p.training,:),T(p.training,:),'Learners',svm);
    labels = predict(Mdlsvm,P(p.test,:));
    errRate_svm_rbf = sum(T(p.test) ~= labels)/p.TestSize;
    conMat_svm_rbf = confusionmat(T(p.test),labels); % the confusion matrix

    classperf(cp_svm_rbf,labels,p.test);
    

    
    %% naive bayers
    naive = templateNaiveBayes();
    CVMdl2 = fitcecoc(P(p.training,:),T(p.training,:),'Learners',naive,'Coding', 'onevsall');    
    labels = predict(CVMdl2,P(p.test,:));
    
    errRate_naive = sum(T(p.test) ~= labels)/p.TestSize;
    conMat_naive = confusionmat(T(p.test),labels); % the confusion matrix

    classperf(cp_naive,labels,p.test);
    
    %% RNA


    XX=features';

    TT=full(ind2vec(target',8));

    %separar dados de treino
    x = XX(:,p.training);
    t = TT(:,p.training);
    %gerar rede neural
    net = patternnet(15);
    net.trainParam.showWindow = false;
    %treinar
    [net,tr] = train(net,x,t);
    %testar
    labels = net(XX(:,p.test));
    errRate_rna = sum(vec2ind(TT(:,p.test)) ~= vec2ind(labels))/p.TestSize;
    conMat_rna = confusionmat(vec2ind(TT(:,p.test)),vec2ind(labels)'); % the confusion matrix

    classperf(cp_rna,vec2ind(labels),p.test);
    
%%
    p = cvpartition(T,'Holdout',0.30);
end

%% plotar resultado

figure;
names = {'SvmLinear ','SvmRbf','SvmQuad','Rna','Naive'};
values = [cp_svm_linear.CorrectRate,cp_svm_rbf.CorrectRate,cp_svm_poly.CorrectRate,cp_rna.CorrectRate,cp_naive.CorrectRate];



for k=1:length(names)
    names{k} = [names{k},' ',num2str(round(values(k)*100,2)),'%'];
end

c = categorical(names);

bar(c,values);
title(['Taxa de acerto Hold-Out 30% (iterações N=',num2str(N),')']);


%% Matriz de confusao
cp=cp_svm_linear;
desenha_matriz_confusao( cp ,emotions_list_note_pt )
title(['Matriz Confusão SVM Linear(iterações N=',num2str(N),', Tx. Acerto ',num2str(cp.CorrectRate*100),'%)']);

cp=cp_svm_poly;
desenha_matriz_confusao( cp ,emotions_list_note_pt )
title(['Matriz Confusão SVM Quadrático(iterações N=',num2str(N),', Tx. Acerto ',num2str(cp.CorrectRate*100),'%)']);

cp=cp_svm_rbf;
desenha_matriz_confusao( cp ,emotions_list_note_pt )
title(['Matriz Confusão SVM RBF(iterações N=',num2str(N),', Tx. Acerto ',num2str(cp.CorrectRate*100),'%)']);

cp=cp_naive;
desenha_matriz_confusao( cp ,emotions_list_note_pt )
title(['Matriz Confusão Naive Bayes(iterações N=',num2str(N),', Tx. Acerto ',num2str(cp.CorrectRate*100),'%)']);

cp=cp_rna;
desenha_matriz_confusao( cp ,emotions_list_note_pt )
title(['Matriz Confusão RNA(iterações N=',num2str(N),', Tx. Acerto ',num2str(cp.CorrectRate*100),'%)']);
%% usar PCA

% emotions_list_note = {'neutral','angry','contemptuous','disgusted','fearful','happy','sad','surprised'};
% emotions_list_class = [1,2,3,4,5,6,7,8];
% map_note = containers.Map(emotions_list_class,emotions_list_note);
% map_class = containers.Map(emotions_list_note,emotions_list_class);
% eg = pca(P,'NumComponents',3);
% 
% PP = P*eg;
% figure;
% color = {'y','m','c','r','g','b','*','k'}
% for k=1:size(emotions_list_class,2)
%    scatter3(PP(find(T==emotions_list_class(k)),1),PP(find(T==emotions_list_class(k)),2),PP(find(T==emotions_list_class(k)),3),color{k});
%    hold on;
%     
%     
% end
% legend({'neutral','angry','contemptuous','disgusted','fearful','happy','sad','surprised'});


%%
Arrhythmia = array2table(features);
Arrhythmia.Class = categorical(target);