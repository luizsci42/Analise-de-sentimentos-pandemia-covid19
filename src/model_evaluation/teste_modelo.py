import pickle
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mglearn.tools import heatmap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def carregar_modelo():
    path = 'resources/modelos/modelo_classificacao_svc'
    try:
        modelo = pickle.load(open(path, 'rb'))
        print('Modelo carregado com sucesso!\n')

        return modelo
    except FileNotFoundError:
        print('O arquivo não foi encontrado')
    except IOError:
        print('Erro ao carregar arquivo')


def testar_performance(df_teste: pd.DataFrame, df_previsoes: pd.DataFrame, exibir_heatmap: bool = False):
    y_teste = list(df_teste['Sentimento'].dropna())
    previsoes = list(df_previsoes['Sentimento'].dropna())
    rotulos_eixos = ['Neutro', 'Feliz', 'Medo', 'Nojo', 'Raiva', 'Triste']

    confusao = confusion_matrix(y_teste, previsoes)

    # print('Matriz de confusão:\n{}'.format(confusao))
    print('Relatório de classificação:\n{}'.format(
        classification_report(y_teste, previsoes)
    ))

    if exibir_heatmap:
        scores = heatmap(
            confusao,
            xlabel='Previsão',
            ylabel='Sentimentos Corretos',
            xticklabels=rotulos_eixos,
            yticklabels=rotulos_eixos
        )
        plt.title('Matriz de Confusão')
        plt.gca().invert_yaxis()
        plt.show()


def fazer_previsoes(alvo: pd.DataFrame, saida_previsoes: str):
    nlp = spacy.load('pt_core_news_lg')
    # textos = ['tô feliz hoje', 'não estou com paciência nenhuma pra nada',
    #          'deveras interessante', 'as coisas são assim mesmo', 'nada me é incômodo hoje',
    #          'sei lá do cabrunco', 'três pratos de tigres', 'sai viado']

    df_tweets = alvo
    textos = list(df_tweets['Texto'])
    modelo = carregar_modelo()
    sentimentos = []

    for texto in textos:
        # print('Frase: ' + texto)
        try:
            # tokenizamos a frase, vetorizamos e colocamos em uma array
            vetor = np.array(nlp(texto).vector)
            # colocamos o vetor no formato 1x300, que é o que o svc espera
            vetor = np.reshape(vetor, (1, 300))
        except TypeError:
            texto = str(texto)
            vetor = np.array(nlp(texto).vector)
            vetor = np.reshape(vetor, (1, 300))
        # fazemos a previsão e retornamos a classe
        previsao = modelo.predict(vetor)[0]
        # print("Sentimento: " + previsao + '\n')
        sentimentos.append(previsao)

    # datas = list(df_tweets['Data'])
    dados = {'Texto': textos, 'Sentimento': sentimentos}
    df_saida = pd.DataFrame(dados)
    df_saida.to_csv(saida_previsoes, mode='w')


if __name__ == '__main__':
    dataset_alvo = pd.read_csv('../../resources/datasets/amostra_dataset.csv')
    path_previsoes = '../../resources/datasets/teste_previsoes.csv'
    fazer_previsoes(dataset_alvo, path_previsoes)

    dataset_previsoes = pd.read_csv('../../resources/datasets/teste_previsoes.csv')
    testar_performance(dataset_alvo, dataset_previsoes, True)
