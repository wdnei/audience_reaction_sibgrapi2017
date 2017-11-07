%% EXTRAIR FACES E SALVAR LBP EM MATRIX
emotions_list_note_pt = {'neutro','raiva','desdenho','nojo','medo','alegria','tristeza','surpresa'};
emotions_list_note = {'neutral','angry','contemptuous','disgusted','fearful','happy','sad','surprised'};
emotions_list_class = [1,2,3,4,5,6,7,8];
map_note = containers.Map(emotions_list_class,emotions_list_note);
map_class = containers.Map(emotions_list_note,emotions_list_class);


dr=dir('../datasets/rafd/*frontal.jpg');

features = zeros(length(dr),59*6);
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
        lbp_features = extrair_lbp_olho_boca(img1);
        if(~isempty(lbp_features))
            features(fc,:) = lbp_features;
            target(fc) = map_class(emotion_note);
        else
            disp(['face nao encontrada:',fullFile]);
        end
        disp(fc);
        
    end
end

save('lbp_olho_boca_3partes_rafd2_frontal.mat','features','target','emotions_list_note','emotions_list_note_pt','emotions_list_class');
disp('Terminou');
