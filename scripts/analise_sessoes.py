
import sys
from pathlib import Path

import pandas as pd
from src.utils.enviroment import get_config
from src.utils import log
from src.utils import plot_cdf

pd.options.display.float_format = '{:_.2f}'.format

def main(
        events_path: str,
        min_session_length: int = 2,
        max_session_length: int = 50):
    
    log("Carregando dados de eventos...")
    # Lê dados do CSV
    df = pd.read_csv(events_path, parse_dates=['event_ts'])
    log(f"Total de eventos carregados: {len(df):_}")
    
    # Agrupa por session_id para calcular estatísticas
    log("Calculando estatísticas por sessão...")
    sessao_stats = df.groupby('session_id').agg(
        inicio_sessao=('event_ts', 'min'),
        fim_sessao=('event_ts', 'max'),
        qtd_eventos=('event_id', 'count'),
        qtd_listings_unicos=('listing_id', 'nunique'),
        # Identifica se é usuário anônimo (user_id é NaN)
        is_anonymous=('user_id', lambda x: x.isna().all())
    )
    
    # Calcula duração das sessões
    sessao_stats['duracao'] = sessao_stats['fim_sessao'] - sessao_stats['inicio_sessao']
    sessao_stats['duracao_segundos'] = sessao_stats['duracao'].dt.total_seconds()
    
    log("\n--- Primeiras 5 sessões ---", True)
    print(sessao_stats.head(5))
    
    # Estatísticas descritivas
    log("\n--- Estatísticas de Eventos por Sessão ---", True)
    stats = sessao_stats['qtd_eventos'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    print(stats)
    
    # Filtra dados para plotagem (remove outliers)
    log(f"\nFiltrando sessões entre {min_session_length} e {max_session_length} eventos...", True)
    df_plot = sessao_stats[sessao_stats['qtd_eventos'] >= min_session_length]
    df_plot = df_plot[df_plot['qtd_eventos'] <= max_session_length]
    log(f"Total de sessões após filtro: {len(df_plot):_}")
    
    # Gera gráfico CDF geral
    log("\nGerando gráfico CDF geral...", True)
    plot_cdf(
        df=df_plot,
        col='qtd_eventos',
        max_limit=max_session_length,
        title='CDF - Distribuição Acumulada de Eventos por Sessão',
        color='#007acc',
        save_path='outputs/cdf/cdf_sessoes_geral.svg'
    )
    
    # Análise por tipo de usuário
    log("\n--- Análise por tipo de usuário ---", True)
    df_anonimos = sessao_stats[sessao_stats['is_anonymous'] == True]
    df_logados = sessao_stats[sessao_stats['is_anonymous'] == False]
    
    log(f"Total de sessões anônimas: {len(df_anonimos):_}")
    log(f"Total de sessões logadas: {len(df_logados):_}")
    
    # Filtra por tamanho mínimo/máximo para cada grupo
    df_anonimos_plot = df_anonimos[
        (df_anonimos['qtd_eventos'] >= min_session_length) &
        (df_anonimos['qtd_eventos'] <= max_session_length)
    ]
    df_logados_plot = df_logados[
        (df_logados['qtd_eventos'] >= min_session_length) &
        (df_logados['qtd_eventos'] <= max_session_length)
    ]
    
    log(f"Sessões anônimas após filtro: {len(df_anonimos_plot):_}")
    log(f"Sessões logadas após filtro: {len(df_logados_plot):_}")
    
    # Estatísticas para usuários anônimos
    log("\n--- Estatísticas - Usuários Anônimos ---", True)
    stats_anonimos = df_anonimos['qtd_eventos'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    print(stats_anonimos)
    
    # Estatísticas para usuários logados
    log("\n--- Estatísticas - Usuários Logados ---", True)
    stats_logados = df_logados['qtd_eventos'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    print(stats_logados)
    
    # CDF para usuários anônimos
    log("\nGerando CDF para usuários anônimos...", True)
    plot_cdf(
        df=df_anonimos_plot,
        col='qtd_eventos',
        max_limit=max_session_length,
        title='CDF - Sessões de Usuários Anônimos',
        color='#ff7f0e',  # Laranja
        save_path='outputs/cdf/cdf_sessoes_anonimos.svg'
    )
    
    # CDF para usuários logados
    log("\nGerando CDF para usuários logados...", True)
    plot_cdf(
        df=df_logados_plot,
        col='qtd_eventos',
        max_limit=max_session_length,
        title='CDF - Sessões de Usuários Logados',
        color='#2ca02c',  # Verde
        save_path='outputs/cdf/cdf_sessoes_logados.svg'
    )
    
    log("\nAnálise concluída! Gráficos salvos em outputs/cdf/")


if __name__ == "__main__":
    config = get_config()

    min_session_length = config['data_preparation']['min_session_length']
    max_session_length = config['data_preparation']['max_session_length']

    main(
        events_path='notebooks/events.csv',
        min_session_length=min_session_length,
        max_session_length=max_session_length
    )
