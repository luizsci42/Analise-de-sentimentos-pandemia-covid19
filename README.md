# Análise de Sentimentos sobre a pandemia de COVID-19 a partir do Twitter
Este repositório tem como finalidade:
* Criar um dataset para treinamento de modelos de aprendizado de máquina sobre as 5 emoções de Ekman.
* Analisar os sentimentos predominantes em determinados períodos da pandemia

## Sobre o dataset de treinamento
O dataset é composto por tweets que contenham 
hashtags relacionadas às 5 emoções de Ekman, capturados em
todo o ano de 2018 e, particularmente para as emoções de 
raiva e nojo, forma feitas capturas exclusivamente sobre essas hashtags
durante todo o ano de 2016 e, ainda devido à insuficiência de amostras,
mais tweets com a hashtag #raiva foram coletados durante todo o ano de 2017.

Os tweets rotulados como neutros foram copiados do Tweets_Mg dataset do
blog minerando dados.

O dataset tweets_ekman.csv, usado para treinamento, possui o segunte formato:

![Proporção de cada emoção no dataset de treiamento](fig/proporcao_dataset.png)


## Sobre o dataset alvo
O dataset tweets_pandemia é composto por tweets capturados entre abril de 2020 e
março de 2021. Os tweets foram selecionados de acordo com a presença das palavras-chave
"isolamento", "quarentena", "covid", "corona", "coronavirus", "corona virus", "covid-19",
"covid19" e/ou "covid 19".


## Sobre o teste do modelo
O modelo de classificação foi feito a partir da implementação do classificador
linear LinearSVC, disponível atrabés do scikit-learn. Esse modelo foi treinado
a partir do dataset tweets_ekman e sseu desempenho foi medido através de uma
matriz de erro (ou matriz de confusão), a partir da qual obtivemos os seguintes
valores:

![Matriz de confusão do modelo](fig/matriz_confusao.png)

As demais medidas de precisão, recall e f-score:

![Relatório de Classificação](fig/relatorio_classificacao.png)

As medidas indicam que devemos revisar o conjunto de treinamento gerado a partir
dos pares tweets e hashtags, já que o modelo está mostrando medidas satisfatórias
para os dados importados do dataset do blog Minerando Dados, mas não para os
dados provenentes do dataset que criamos. Um bom ponto de partida é reler o artigo
de (Niko e Demšar, 2018), para ver como foi feito o tratamento dos tweets no
trabalho deles, analisar se é aplicável ao nosso contexto e, caso positivo, tentar
reproduzir.

## Referências
Karami, Amir, et al. "Twitter and research: a systematic
literature review through text mining." IEEE Access 8 (2020):
67698-67717.

P. Ekman, “An argument for basic emotions,” Cognition Emotion,
vol. 6, no. 3, pp. 169–200, 1992.

Colnerič, Niko, and Janez Demšar. "Emotion recognition on 
twitter: Comparative study and training a unison model." IEEE 
transactions on affective computing 11.3 (2018): 433-446.