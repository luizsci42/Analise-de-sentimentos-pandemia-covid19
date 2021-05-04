import spacy
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

"""
Alguns artigos que podem ajudar para resolver a falta de acurácia do modelo:
Towards Data Science - How to Deal with Imbalanced Data: https://towardsdatascience.com/how-to-deal-with-imbalanced-data-34ab7db9b100
KDNuggets - 7 Techniques to Handle Imbalanced Data: https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
ML+ - How to Train Text Classification in SpaCy: https://www.machinelearningplus.com/nlp/custom-text-classification-spacy/
Advanced NLP with SpaCy - Chapter 4: Training a neural network model: https://course.spacy.io/en/chapter4
"""


def ler_conjunto_treinamento(path: str):
    """
    Lê um arquivo .csv e retorna um Pandas Dataframe com vários tweets rotulados em três polaridades:
    positivo, negativo e neutro.

    :return: Pandas Dataframe
    """

    return pd.read_csv(path).dropna()


def fazer_amostragem(train_dataset: pd.DataFrame):
    """

    :param train_dataset:
    :return:
    """

    df_neutro = train_dataset.loc[train_dataset['Sentimento'] == 'Neutro'][:600]
    df_feliz = train_dataset.loc[train_dataset['Sentimento'] == 'feliz'][:600]
    df_medo = train_dataset.loc[train_dataset['Sentimento'] == 'medo'][:600]
    df_nojo = train_dataset.loc[train_dataset['Sentimento'] == 'nojo'][:600]
    df_raiva = train_dataset.loc[train_dataset['Sentimento'] == 'raiva'][:600]
    df_triste = train_dataset.loc[train_dataset['Sentimento'] == 'triste'][:600]

    df_novo = df_neutro.append(df_feliz)
    df_novo = df_novo.append(df_medo)
    df_novo = df_novo.append(df_nojo)
    df_novo = df_novo.append(df_raiva)
    df_novo = df_novo.append(df_triste)

    df_novo.to_csv('datasets/amostra_dataset.csv')

    return df_novo


def treinar_modelo(df_tweets, vetores):
    """
    Utiliza o Spacy para vetorizar o texto dos tweets em word embeddings
    :return:
    """
    # separamos os dados em conjuntos de treinamento e de teste
    print('Separando conjuntos de teste e treinamento')
    X_train, X_test, y_train, y_test = train_test_split(vetores, df_tweets.Sentimento, test_size=0.1, random_state=0)
    # selecionamos o modelo de aprendizado de máquina com algumas configurações
    svc = LinearSVC(C=100, random_state=0, dual=True, max_iter=10000)
    # mlp = MLPClassifier(random_state=0).fit(X_train, y_train)
    # treinamos o modelo
    print('Treinando modelo...')
    svc.fit(X_train, y_train)
    print('Modelo treinado com êxito!')
    # 0.59% para C = 1 | 0.59% para C = 1000  | 0.59% para C = 10000
    # print('Acurácia no dataset de treinamento: {acc:.2f}%'.format(acc=svc.score(X_train, y_train)))
    # 58.81% para C = 1 | 58.86% para C = 1000 | 58.88% para C = 10000 | caiu para 42% ao aicionar os neutros
    # print('Acurácia no dataset de teste: {acc:.2f}%'.format(acc=svc.score(X_test, y_test) * 100))

    return svc


def criar_embeddings(df_tweets):
    nlp = spacy.load('pt_core_news_lg')
    # desativamos todos os outros pipes que vem com o modelo nlp porque não preicsaremos deles
    with nlp.disable_pipes():
        # transformamos cada texto em um vetor e colocamos em uma array
        print('Fazendo os word embeddings')
        vetores = np.array([nlp(texto).vector for texto in df_tweets.Texto])

    return vetores


def salvar_modelo(modelo):
    path = 'modelos/modelo_classificacao_svc_100'
    try:
        pickle.dump(modelo, open(path, 'wb'))
        print("Modelo salvo com sucesso!")
    except IOError:
        print("Erro ao salvar modelo")


def salvar_embeddings(embeddings):
    path = 'modelos/embeddings'
    try:
        pickle.dump(embeddings, open(path, 'wb'))
        print("Embeddings salvo com sucesso!")
    except IOError:
        print("Erro ao salvar embeddings")


def ler_embeddings():
    path = 'modelos/embeddings'
    try:
        embeddings = pickle.load(open(path, 'rb'))
        print('Embeddings carregado com sucesso!\n')

        return embeddings
    except FileNotFoundError:
        print('O arquivo não foi encontrado')
    except IOError:
        print('Erro ao carregar arquivo')


def main():
    path = 'datasets/tweets_ekman.csv'
    df_tweets = ler_conjunto_treinamento(path)
    df_tweets = fazer_amostragem(df_tweets)

    # vetores = criar_embeddings(df_tweets)
    # salvar_embeddings(vetores)

    embeddings = ler_embeddings()
    modelo = treinar_modelo(df_tweets, embeddings)
    salvar_modelo(modelo)


if __name__ == '__main__':
    main()
