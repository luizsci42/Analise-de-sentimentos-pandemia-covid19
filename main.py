import pandas as pd
from src.model_evaluation.teste_modelo import fazer_previsoes

tweets_pandemia = pd.read_csv('resources/datasets/tweets_pandemia.csv')
path_previsoes = 'resources/datasets/previsoes_no_dataset_alvo.csv'
fazer_previsoes(tweets_pandemia, path_previsoes)
