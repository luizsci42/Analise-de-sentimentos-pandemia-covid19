import spacy
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


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


def treinar_modelo(df_tweets: pd.DataFrame, vetores: np.array, modelo):
    """
    Treinamos o modelo de aprendizado de máquina para fazer a análise de sentimentos.
    O texto está codificado em word embeddings. Já os rótulos, estão em formato comum
    de texto.

    :param df_tweets: O dataset contendo os rótulos.
    :param vetores: As word embeddings que representam o texto a ser rotulado.
    :param modelo: O modelo que será treinado.
    :return: O modelo treinado.
    """
    # separamos os dados em conjuntos de treinamento e de teste
    print('Separando conjuntos de teste e treinamento')
    x_train, x_test, y_train, y_test = train_test_split(vetores, df_tweets.Sentimento, test_size=0.1, random_state=0)
    # treinamos o modelo
    print('Treinando modelo...')
    modelo.fit(x_train, y_train)
    print('Modelo treinado com êxito!')

    return modelo


def criar_embeddings(df_tweets):
    nlp = spacy.load('pt_core_news_lg')
    # desativamos todos os outros pipes que vem com o modelo nlp porque não preicsaremos deles
    with nlp.disable_pipes():
        # transformamos cada texto em um vetor e colocamos em uma array
        print('Fazendo os word embeddings')
        vetores = np.array([nlp(texto).vector for texto in df_tweets.Texto])

    return vetores


def salvar_modelo(modelo, nome_arquivo: str):
    try:
        pickle.dump(modelo, open(nome_arquivo, 'wb'))
        print("Modelo salvo com sucesso!")
    except IOError:
        print("Erro ao salvar modelo")


def salvar_embeddings(embeddings):
    path = '../../resources/modelos/embeddings'
    try:
        pickle.dump(embeddings, open(path, 'wb'))
        print("Embeddings salvo com sucesso!")
    except IOError:
        print("Erro ao salvar embeddings")


def ler_embeddings():
    path = '../../resources/modelos/embeddings'
    try:
        embeddings = pickle.load(open(path, 'rb'))
        print('Embeddings carregado com sucesso!\n')

        return embeddings
    except FileNotFoundError:
        print('O arquivo não foi encontrado')
    except IOError:
        print('Erro ao carregar arquivo')


def main():
    path = '../../resources/datasets/tweets_ekman.csv'
    df_tweets = ler_conjunto_treinamento(path)
    df_tweets = fazer_amostragem(df_tweets)

    # vetores = criar_embeddings(df_tweets)
    # salvar_embeddings(vetores)

    embeddings = ler_embeddings()
    # selecionamos o modelo de aprendizado de máquina com algumas configurações
    svc = LinearSVC(C=100, random_state=0, dual=True, max_iter=10000)
    # também poderíamos usar outro modelo, como o MLPClassifier
    # mlp = MLPClassifier(random_state=0).fit(X_train, y_train)
    modelo = treinar_modelo(df_tweets, embeddings, svc)

    path = 'resources/modelos/modelo_classificacao_svc_100'
    salvar_modelo(modelo, path)


if __name__ == '__main__':
    main()
