import pandas as pd
from teste_modelo import fazer_previsoes

tweets_pandemia = pd.read_csv('datasets/tweets_pandemia.csv')
path_previsoes = 'datasets/previsoes_no_dataset_alvo.csv'
fazer_previsoes(tweets_pandemia, path_previsoes)
