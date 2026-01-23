import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np


def plot_cdf(df,
                     col='qtd_eventos',
                     max_limit=30,
                     title='CDF - Distribuição Acumulada',
                     color='#007acc',
                     save_path=None):
    """
    Gera um gráfico CDF (Cumulative Distribution Function) para análise de sessões.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        col (str): Nome da coluna a ser analisada.
        max_limit (int): Valor máximo do eixo X para corte visual (ex: 30 eventos).
        title (str): Título do gráfico.
        color (str): Cor da linha do gráfico.
        save_path (str, optional): Caminho para salvar a imagem (ex: 'grafico.png'). Se None, apenas exibe.
    """

    df_plot = df[df[col] <= max_limit].copy()
    plt.figure(figsize=(12, 6))

    # Plota a CDF
    sns.ecdfplot(data=df_plot, x=col, color=color, linewidth=2)

    # Configurações de Texto
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Quantidade de Eventos (Até X)', fontsize=12)
    plt.ylabel('Proporção Acumulada de Sessões (%)', fontsize=12)

    # Configurações de Eixos
    step = 1
    plt.xticks(np.arange(0, max_limit + 1, step))
    # Formata Eixo Y para porcentagem
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    # Linhas de Referência
    percentiles = [0.25, 0.50, 0.75, 0.90]
    for p in percentiles:
        plt.axhline(y=p, color='gray', linestyle=':', alpha=0.5)
        # O texto fica levemente acima da linha (p + 0.01)
        plt.text(0, p + 0.01, f'{int(p * 100)}% das sessões', color='gray', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()

    plt.close()