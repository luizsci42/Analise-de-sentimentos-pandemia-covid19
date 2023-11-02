import pandas as pd
from src.model_evaluation.teste_modelo import fazer_previsoes, testar_performance

tweets_pandemia = pd.read_csv('resources/datasets/tweets_pandemia.csv')
path_previsoes = 'resources/datasets/previsoes_no_dataset_alvo.csv'
fazer_previsoes(tweets_pandemia, path_previsoes)

dataset_previsoes = pd.read_csv('resources/datasets/previsoes_no_dataset_alvo.csv')
testar_performance(tweets_pandemia, dataset_previsoes, True)
