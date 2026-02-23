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


    df_plot = df[df[col] <= max_limit].copy()
    plt.figure(figsize=(12, 6))


    sns.ecdfplot(data=df_plot, x=col, color=color, linewidth=2)


    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Quantidade de Eventos', fontsize=12)
    plt.ylabel('Proporção Acumulada de Sessões (%)', fontsize=12)


    step = 1
    plt.xticks(np.arange(0, max_limit + 1, step))

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(True, which='both', linestyle='--', alpha=0.6)


    percentiles = [0.25, 0.50, 0.75, 0.90]
    for p in percentiles:
        plt.axhline(y=p, color='gray', linestyle=':', alpha=0.5)

        plt.text(0, p + 0.01, f'{int(p * 100)}% das sessões', color='gray', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()

    plt.close()
