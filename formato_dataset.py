import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

arquivo = 'datasets/tweets_ekman.csv'
df_ekman_tt = pd.read_csv(arquivo)

rotulos = ['Feliz', 'Medo', 'Raiva', 'Nojo', 'Triste', 'Neutro']
sentimentos = list(df_ekman_tt.Sentimento)
valores = [sentimentos.count('feliz'), sentimentos.count('medo'), sentimentos.count('raiva'),
           sentimentos.count('nojo'), sentimentos.count('triste'), sentimentos.count('Neutro')]

sns.set()
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('equal')
ax.set_title('Proporção dos sentimentos no dataset de treinamento')
ax.pie(valores, labels=rotulos, autopct='%1.2f%%')

fig.savefig('fig/proporcao_dataset.png')
print('Feliz: {}\nMedo: {}\nRaiva: {}\nNojo: {}\nTriste: {}\nNeutro: {}'.format(
    valores[0], valores[1], valores[2], valores[3], valores[4], valores[5]))
