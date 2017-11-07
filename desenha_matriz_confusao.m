function [ output_args ] = desenha_matriz_confusao( matriz_confusao,classe_nomes )
%DESENHA_MATRIZ_CONFUSAO Summary of this function goes here
%   Detailed explanation goes here
figure;

linhaTexto='Classe Verdadeira';
colunaTexto='Classe Predita';

if(isa(matriz_confusao,'biolearning.classperformance'))
    mat = matriz_confusao.CountingMatrix(1:end-1,:); % usando classperf remove ultima linha
    linhaTexto='Classe Predita';
    colunaTexto='Classe Verdadeira';
else
    mat = matriz_confusao;
end


           %# A 5-by-5 matrix of random values from 0 to 1
imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:length(classe_nomes));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:length(classe_nomes),...                         %# Change the axes tick marks
        'XTickLabel',classe_nomes,...  %#   and tick labels
        'YTick',1:length(classe_nomes),...
        'YTickLabel',classe_nomes,...
        'TickLength',[0 0]);
ylabel(linhaTexto);
xlabel(colunaTexto);
end

