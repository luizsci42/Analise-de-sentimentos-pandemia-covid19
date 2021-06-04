from Scweet.Scweet.scweet import scrap
import pandas as pd
import re

"""
Posso usar multithreading ou multi-processamento para fazer várias requisições ao Twitter simultâneamente.
Um tutorial de multithreading está disponível em: https://www.tutorialspoint.com/python/python_multithreading.htm
ou na documentação oficial: https://docs.python.org/3/tutorial/stdlib2.html#multi-threading

Outros links que podem ser úteis:
Python Library Reference - Thread-based paralelism: https://docs.python.org/3/library/threading.html

"""


class ConsultorTwitter:
    def obter_tweets_por_hashtag(self, hashtag, data_inicial, data_final):
        dados = scrap(hashtag=hashtag, start_date=data_inicial, max_date=data_final, from_account=None, interval=1,
                     headless=True, display_type="Top", save_images=False,
                     resume=False, filter_replies=True, proximity=False, lang='pt')

        return list(dados['Text'])

    def obter_tweets_por_topico(self, topicos, data_inicial, data_final):
        dados = scrap(words=topicos, start_date=data_inicial, max_date=data_final, from_account=None, interval=1,
                     headless=True, display_type="Top", save_images=False,
                     resume=False, filter_replies=True, proximity=False, lang='pt')

        return dados

    def limpar_texto(self, texto):
        texto = texto.lower()
        texto = re.sub('\n', '', texto)
        texto = re.sub('@\S+', '', texto)
        texto = re.sub(r'https://\S+', '', texto)

        return texto

    def tratar_hashtags(self, tweet):
        """
        Usamos expressões regulares para separar a hashtag e
        colocá-la como rótulo do texto
        """
        emocoes = ['#feliz', '#medo', '#raiva', '#nojo', '#triste']
        texto = self.limpar_texto(tweet)
        texto = re.sub(r'#\S+', '', texto)
        hashtags = re.findall('#\S+', tweet)
        sentimento = [sentimento for sentimento in hashtags if sentimento in emocoes]

        if not sentimento:
            return texto, ''
        else:
            sentimento = re.sub('#', '', sentimento[0])
            return texto, sentimento

    def exportar_tweets(self, dados, nome_arquivo):
        """
        Limpa cada tweet e os exporta em formato CSV.

        :param nome_arquivo:
        :param dados: Um dicionário com os dados dos tweets.
        :return:
        """
        df_tweets = pd.DataFrame(dados, columns=['Texto', 'Data'])
        path = '../../resources/datasets/'
        df_tweets.to_csv(path_or_buf=path + nome_arquivo, mode='w')

        return df_tweets


def buscar_hashtags():
    topicos = ['nojo']
    data_inicial = '2017-12-06'
    data_final = '2017-12-31'
    cnsltr_emocoes = ConsultorTwitter()

    resultados = []
    for topico in topicos:
        # Uma lista com o texto de cada tweet
        res = cnsltr_emocoes.obter_tweets_por_hashtag(topico, data_inicial, data_final)
        resultados.append(res)

    tweets = []
    for resultado in resultados:
        for tweet in resultado:
            txt_sentimento = cnsltr_emocoes.tratar_hashtags(tweet)
            tweets.append(txt_sentimento)

    cnsltr_emocoes.exportar_tweets(tweets, topicos[0] + '_tweets_ekman_' + data_inicial + '.csv')


def buscar_topicos():
    topicos = ['isolamento', 'quarentena', 'covid', 'corona', 'coronavírus', 'corona vírus', 'covid-19', 'covid19', 'covid 19']
    data_inicial = '2020-04-01'
    data_final = '2020-04-02'
    cnsltr_emocoes = ConsultorTwitter()

    resultados = []
    # cada resultado é um DataFrame com colunas 'Text' e 'Timestamp'
    res = cnsltr_emocoes.obter_tweets_por_topico(topicos, data_inicial, data_final)
    print(res.head())
    """
    resultados.append(res['Text'])

    tweets = []
    for tweet in resultados:
        texto = cnsltr_emocoes.limpar_texto(tweet)
        tweets.append(texto)

    nome_arquivo = 'pandemia_{}_{}.csv'.format(data_inicial, data_final)
    cnsltr_emocoes.exportar_tweets(tweets, nome_arquivo)
    """


if __name__ == '__main__':
   buscar_topicos()
