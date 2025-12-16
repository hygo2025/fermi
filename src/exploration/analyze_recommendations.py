"""
Análise qualitativa de recomendações de modelos sequenciais.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem display para ambientes SSH
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import traceback
from typing import Dict, List, Any

# Adicionar diretório do projeto ao PATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.exploration.model_explorer import ModelExplorer

# Configurar estilo dos gráficos
plt.style.use('ggplot')
sns.set_palette("husl")


def print_section_header(title: str, char: str = "=", width: int = 100):
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def analyze_single_session(
    explorer: ModelExplorer,
    session_items: List[int],
    session_id: str = "Unknown",
    top_k: int = 10
) -> Dict[str, Any]:
    print(f"\nAnalisando Sessao: {session_id}")
    print(f"  Itens: {session_items}")
    
    result = {
        'session_id': session_id,
        'session_size': len(session_items)
    }
    
    # Caracterizar sessão
    if explorer.item_features is not None:
        sess_df = explorer.item_features[
            explorer.item_features['listing_id'].isin(session_items)
        ]
        
        if len(sess_df) > 0:
            print(f"\n  {len(sess_df)} itens encontrados nas features")
            
            if 'price' in sess_df.columns:
                prices = sess_df['price'].dropna()
                if len(prices) > 0:
                    avg_price = prices.mean()
                    result['session_avg_price'] = avg_price
                    print(f"  Preco medio: R$ {avg_price:,.0f}")
                    print(f"  Faixa: R$ {prices.min():,.0f} - R$ {prices.max():,.0f}")
                    
                    if avg_price > 1000000:
                        profile = "LUXO"
                    elif avg_price < 300000:
                        profile = "ECONOMICO"
                    else:
                        profile = "PADRAO"
                    result['profile'] = profile
                    print(f"  Perfil: {profile}")
            
            if 'city' in sess_df.columns:
                cities = sess_df['city'].value_counts().to_dict()
                result['session_cities'] = cities
                print(f"  Cidades: {cities}")
            
            if 'bedrooms' in sess_df.columns:
                bedrooms = sess_df['bedrooms'].value_counts().to_dict()
                result['session_bedrooms'] = bedrooms
                print(f"  Quartos: {bedrooms}")
    
    print(f"\n  Gerando Top-{top_k} recomendacoes...")
    try:
        recs = explorer.recommend_for_session(session_items, top_k=top_k)
        
        if not recs or len(recs) == 0:
            print("  AVISO: Nenhuma recomendacao gerada")
            result['error'] = 'no_recommendations'
            return result
        
        result['recommendations'] = recs
        print(f"  Recomendacoes: {recs}")
        
        # Comparar características
        if explorer.item_features is not None and len(sess_df) > 0:
            rec_df = explorer.item_features[
                explorer.item_features['listing_id'].isin(recs)
            ]
            
            if len(rec_df) > 0:
                print(f"\n  Comparando caracteristicas...")
                
                if 'price' in sess_df.columns and 'price' in rec_df.columns:
                    sess_prices = sess_df['price'].dropna()
                    rec_prices = rec_df['price'].dropna()
                    
                    if len(sess_prices) > 0 and len(rec_prices) > 0:
                        sess_price = sess_prices.mean()
                        rec_price = rec_prices.mean()
                        diff_pct = ((rec_price - sess_price) / sess_price * 100)
                        
                        result['rec_avg_price'] = rec_price
                        result['price_diff_pct'] = diff_pct
                        
                        print(f"  Preco sessao: R$ {sess_price:,.0f}")
                        print(f"  Preco recomendacoes: R$ {rec_price:,.0f}")
                        print(f"  Diferenca: {diff_pct:+.1f}%", end="")
                        
                        if abs(diff_pct) < 20:
                            print(" [SIMILAR]")
                            result['price_match'] = 'SIMILAR'
                        elif abs(diff_pct) < 50:
                            print(" [MODERADO]")
                            result['price_match'] = 'MODERADO'
                        else:
                            print(" [GRANDE]")
                            result['price_match'] = 'GRANDE'
                
                if 'city' in sess_df.columns and 'city' in rec_df.columns:
                    sess_cities = set(sess_df['city'].dropna())
                    rec_cities = set(rec_df['city'].dropna())
                    overlap = sess_cities & rec_cities
                    
                    result['rec_cities'] = list(rec_cities)
                    
                    print(f"\n  Cidades sessao: {sess_cities}")
                    print(f"  Cidades recomendacoes: {rec_cities}")
                    
                    if len(overlap) > 0:
                        print(f"  Overlap: {overlap}")
                        result['city_match'] = 'SIM'
                    else:
                        print(f"  Sem overlap")
                        result['city_match'] = 'NAO'
                
                if 'bedrooms' in sess_df.columns and 'bedrooms' in rec_df.columns:
                    sess_bedrooms = sess_df['bedrooms'].dropna().median()
                    rec_bedrooms = rec_df['bedrooms'].dropna().median()
                    
                    result['session_bedrooms_median'] = sess_bedrooms
                    result['rec_bedrooms_median'] = rec_bedrooms
                    
                    print(f"\n  Quartos sessao (mediana): {sess_bedrooms}")
                    print(f"  Quartos recomendacoes (mediana): {rec_bedrooms}")
                    
                    if sess_bedrooms == rec_bedrooms:
                        print(f"  Match: IGUAL")
                        result['bedrooms_match'] = 'IGUAL'
                    else:
                        diff = abs(sess_bedrooms - rec_bedrooms)
                        if diff <= 1:
                            print(f"  Match: PROXIMO ({diff:.0f})")
                            result['bedrooms_match'] = 'PROXIMO'
                        else:
                            print(f"  Match: DIFERENTE ({diff:.0f})")
                            result['bedrooms_match'] = 'DIFERENTE'
            else:
                print("  Recomendacoes sem features disponiveis")
        else:
            print("  Sessao sem features disponiveis")
    
    except Exception as e:
        print(f"  Erro ao gerar recomendacoes: {e}")
        result['error'] = str(e)
    
    return result


def analyze_multiple_sessions(
    explorer: ModelExplorer,
    test_df: pd.DataFrame,
    num_sessions: int = 5,
    top_k: int = 5
) -> pd.DataFrame:
    print_section_header("ANALISE DE MULTIPLAS SESSOES")
    
    valid_sessions_df = test_df.groupby('session_id:token').filter(lambda x: len(x) >= 3)
    unique_sessions = valid_sessions_df['session_id:token'].unique()
    
    print(f"\nTotal de sessoes validas no dataset: {len(unique_sessions):,}")
    print(f"Analisando {num_sessions} sessoes aleatorias...\n")
    
    # Selecionar sessões aleatoriamente
    selected_sessions = np.random.choice(
        unique_sessions,
        size=min(num_sessions, len(unique_sessions)),
        replace=False
    )
    
    results = []
    
    for i, session_id in enumerate(selected_sessions, 1):
        print_section_header(f"SESSÃO {i}/{len(selected_sessions)}: {session_id}", char="-")
        
        try:
            # Extrair itens da sessão
            session_data = test_df[
                test_df['session_id:token'] == session_id
            ].sort_values('timestamp:float')
            
            session_items = session_data['item_id:token'].tolist()[:5]
            
            if not session_items:
                print("AVISO: Sessao vazia, pulando...")
                results.append({
                    'session_id': session_id,
                    'error': 'empty_session'
                })
                continue
            
            # Analisar sessão
            result = analyze_single_session(
                explorer,
                session_items,
                session_id,
                top_k
            )
            results.append(result)
        
        except Exception as e:
            print(f"\nErro ao processar sessao: {e}")
            traceback.print_exc()
            results.append({
                'session_id': session_id,
                'error': str(e)
            })
    
    results_df = pd.DataFrame(results)
    
    print_section_header("RESUMO ESTATISTICO")
    
    # Filtrar sessões com erro para estatísticas
    if 'error' in results_df.columns:
        errors_count = results_df['error'].notna().sum()
        if errors_count > 0:
            print(f"\nAVISO: {errors_count} sessoes com erro foram ignoradas nas estatisticas")
        results_df_clean = results_df[results_df['error'].isna()].copy()
    else:
        results_df_clean = results_df.copy()
    
    if len(results_df_clean) == 0:
        print("\nERRO: Nenhuma sessao valida para analise")
        return results_df
    
    print(f"\nResultados por sessao ({len(results_df_clean)} validas):")
    display_cols = [
        'session_id', 'profile', 'price_match', 
        'city_match', 'bedrooms_match'
    ]
    available_cols = [col for col in display_cols if col in results_df_clean.columns]
    if available_cols:
        print(results_df_clean[available_cols].to_string(index=False))
    else:
        print(results_df_clean.to_string(index=False))
    
    if 'price_match' in results_df_clean.columns:
        print("\nCompatibilidade de Precos:")
        price_counts = results_df_clean['price_match'].value_counts()
        for match, count in price_counts.items():
            pct = (count / len(results_df_clean)) * 100
            print(f"  {match}: {count} ({pct:.1f}%)")
    
    if 'city_match' in results_df_clean.columns:
        print("\nCompatibilidade de Localizacao:")
        city_match_pct = (results_df_clean['city_match'] == 'SIM').sum() / len(results_df_clean) * 100
        print(f"  Sessoes com overlap de cidade: {city_match_pct:.1f}%")
    
    if 'bedrooms_match' in results_df_clean.columns:
        print("\nCompatibilidade de Quartos:")
        bedrooms_counts = results_df_clean['bedrooms_match'].value_counts()
        for match, count in bedrooms_counts.items():
            pct = (count / len(results_df_clean)) * 100
            print(f"  {match}: {count} ({pct:.1f}%)")
    
    if 'profile' in results_df_clean.columns:
        print("\nPerfis de Sessoes:")
        profile_counts = results_df_clean['profile'].value_counts()
        for profile, count in profile_counts.items():
            pct = (count / len(results_df_clean)) * 100
            print(f"  {profile}: {count} ({pct:.1f}%)")
    
    return results_df


def create_visualizations(
    explorer: ModelExplorer,
    session_items: List[int],
    top_k: int = 10,
    output_dir: Path = None
):
    print_section_header("CRIANDO VISUALIZACOES")
    
    if explorer.item_features is None:
        print("Features nao disponiveis para visualizacao")
        return
    
    # Obter dados
    sess_df = explorer.item_features[
        explorer.item_features['listing_id'].isin(session_items)
    ]
    
    try:
        recs = explorer.recommend_for_session(session_items, top_k=top_k)
        
        if not recs or len(recs) == 0:
            print("Nenhuma recomendacao gerada, pulando visualizacoes")
            return
        
        rec_df = explorer.item_features[
            explorer.item_features['listing_id'].isin(recs)
        ]
        
        if len(sess_df) == 0 or len(rec_df) == 0:
            print("Dados insuficientes para visualizacao")
            return
    except Exception as e:
        print(f"Erro ao gerar recomendacoes para visualizacao: {e}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Analise Comparativa: Sessao vs Recomendacoes', 
                 fontsize=16, fontweight='bold')
    
    # 1. Histograma de preços
    if 'price' in sess_df.columns and 'price' in rec_df.columns:
        sess_prices = sess_df['price'].dropna()
        rec_prices = rec_df['price'].dropna()
        
        if len(sess_prices) > 0 and len(rec_prices) > 0:
            axes[0].hist(sess_prices, bins=10, alpha=0.6, 
                        label='Sessao', color='blue', edgecolor='black')
            axes[0].hist(rec_prices, bins=10, alpha=0.6, 
                        label='Recomendacoes', color='red', edgecolor='black')
            
            sess_mean = sess_prices.mean()
            rec_mean = rec_prices.mean()
            axes[0].axvline(sess_mean, color='blue', linestyle='--', 
                           linewidth=2, label=f'Media sessao: R$ {sess_mean:,.0f}')
            axes[0].axvline(rec_mean, color='red', linestyle='--', 
                           linewidth=2, label=f'Media recs: R$ {rec_mean:,.0f}')
            
            axes[0].set_xlabel('Preco (R$)', fontsize=12)
            axes[0].set_ylabel('Frequencia', fontsize=12)
            axes[0].set_title('Distribuicao de Precos', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
    
    # 2. Boxplot de áreas
    area_col = 'usable_areas' if 'usable_areas' in rec_df.columns else 'total_areas'
    if area_col in sess_df.columns and area_col in rec_df.columns:
        sess_areas = sess_df[area_col].dropna()
        rec_areas = rec_df[area_col].dropna()
        
        if len(sess_areas) > 0 and len(rec_areas) > 0:
            data_to_plot = [sess_areas, rec_areas]
            bp = axes[1].boxplot(data_to_plot, labels=['Sessao', 'Recomendacoes'],
                                patch_artist=True, widths=0.6)
            
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            axes[1].set_ylabel('Area (m2)', fontsize=12)
            axes[1].set_title('Distribuicao de Areas', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            sess_median = sess_areas.median()
            rec_median = rec_areas.median()
            axes[1].text(1, sess_median, f'{sess_median:.0f}m2', 
                        ha='left', va='center', fontweight='bold')
            axes[1].text(2, rec_median, f'{rec_median:.0f}m2', 
                        ha='left', va='center', fontweight='bold')
    
    if 'bedrooms' in sess_df.columns and 'bedrooms' in rec_df.columns:
        sess_bedrooms = sess_df['bedrooms'].value_counts().sort_index()
        rec_bedrooms = rec_df['bedrooms'].value_counts().sort_index()
        
        all_bedrooms = sorted(set(sess_bedrooms.index) | set(rec_bedrooms.index))
        
        x = np.arange(len(all_bedrooms))
        width = 0.35
        
        sess_counts = [sess_bedrooms.get(b, 0) for b in all_bedrooms]
        rec_counts = [rec_bedrooms.get(b, 0) for b in all_bedrooms]
        
        axes[2].bar(x - width/2, sess_counts, width, 
                   label='Sessao', color='blue', alpha=0.7)
        axes[2].bar(x + width/2, rec_counts, width, 
                   label='Recomendacoes', color='red', alpha=0.7)
        
        axes[2].set_xlabel('Numero de Quartos', fontsize=12)
        axes[2].set_ylabel('Frequencia', fontsize=12)
        axes[2].set_title('Distribuicao de Quartos', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f'{int(b)}' for b in all_bedrooms])
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / 'comparative_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nGrafico salvo em: {fig_path}")
    else:
        print("\nAVISO: output_dir nao especificado, grafico nao foi salvo")
    
    plt.close(fig)


def print_final_summary(results_df: pd.DataFrame):
    print_section_header("RESUMO FINAL")
    
    # Filtrar erros
    if 'error' in results_df.columns:
        results_df = results_df[results_df['error'].isna()].copy()
    
    if len(results_df) == 0:
        print("\nNenhuma sessao valida para gerar resumo")
        return
    
    print("\nCHECKLIST DE VALIDACAO:")
    print("  [ ] As recomendacoes tem precos similares a sessao?")
    print("  [ ] As recomendacoes sao da mesma regiao geografica?")
    print("  [ ] O numero de quartos/tamanho e compativel?")
    print("  [ ] O tipo de imovel faz sentido?")
    print("  [ ] O modelo se adapta a diferentes perfis?")
    print("  [ ] Ha diversidade nas recomendacoes?")
    
    print("\nVALIDACAO AUTOMATICA:")
    
    if 'price_match' in results_df.columns:
        similar_pct = (results_df['price_match'] == 'SIMILAR').sum() / len(results_df) * 100
        if similar_pct >= 60:
            status = "BOM"
        elif similar_pct >= 30:
            status = "MODERADO"
        else:
            status = "RUIM"
        print(f"  Precos similares: {similar_pct:.1f}% [{status}]")
    
    if 'city_match' in results_df.columns:
        city_pct = (results_df['city_match'] == 'SIM').sum() / len(results_df) * 100
        if city_pct >= 60:
            status = "BOM"
        elif city_pct >= 30:
            status = "MODERADO"
        else:
            status = "RUIM"
        print(f"  Localizacao compativel: {city_pct:.1f}% [{status}]")
    
    if 'bedrooms_match' in results_df.columns:
        bedrooms_ok = (results_df['bedrooms_match'].isin(['IGUAL', 'PROXIMO'])).sum()
        bedrooms_pct = bedrooms_ok / len(results_df) * 100
        if bedrooms_pct >= 60:
            status = "BOM"
        elif bedrooms_pct >= 30:
            status = "MODERADO"
        else:
            status = "RUIM"
        print(f"  Quartos compativeis: {bedrooms_pct:.1f}% [{status}]")
    
    print("\nPROXIMOS PASSOS:")
    print("  1. Revisar resultados do checklist acima")
    print("  2. Se necessario, ajustar hiperparametros e retreinar")
    print("  3. Validar com mais sessoes de teste")
    print("  4. Apresentar resultados para stakeholders")
    print("  5. Considerar deploy em producao (A/B test)")


def main():
    parser = argparse.ArgumentParser(
        description='Analise de recomendacoes de modelos sequenciais'
    )
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--num_sessions', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--no_viz', action='store_true')
    
    args = parser.parse_args()
    
    print_section_header("ANALISE DE RECOMENDACOES", char="=")
    print(f"\nModelo: {args.model_path}")
    print(f"Features: {args.features_path}")
    print(f"Test Data: {args.test_data_path}")
    print(f"Sessoes a analisar: {args.num_sessions}")
    print(f"Top-K: {args.top_k}")
    
    try:
        print_section_header("CARREGANDO MODELO")
        explorer = ModelExplorer(args.model_path)
        print("Modelo carregado com sucesso!")
        print(f"Total de itens: {explorer.dataset.item_num:,}")
        
        print_section_header("CARREGANDO FEATURES")
        explorer.load_item_features(args.features_path)
        print("Features carregadas com sucesso!")
        print(f"Total de anuncios: {len(explorer.item_features):,}")
        
        print_section_header("CARREGANDO DADOS DE TESTE")
        test_df = pd.read_csv(args.test_data_path, sep='\t')
        print("Dados de teste carregados com sucesso!")
        print(f"Total de interacoes: {len(test_df):,}")
        print(f"Colunas: {test_df.columns.tolist()}")
        
        print_section_header("ANALISE DE SESSAO EXEMPLO")
        valid_sessions = test_df.groupby('session_id:token').filter(lambda x: len(x) >= 3)
        first_session_id = valid_sessions['session_id:token'].iloc[0]
        first_session_data = test_df[
            test_df['session_id:token'] == first_session_id
        ].sort_values('timestamp:float')
        first_session_items = first_session_data['item_id:token'].tolist()[:5]
        
        analyze_single_session(
            explorer,
            first_session_items,
            first_session_id,
            args.top_k
        )
        
        results_df = analyze_multiple_sessions(
            explorer,
            test_df,
            args.num_sessions,
            top_k=5
        )
        
        if not args.no_viz:
            output_dir = Path(args.output_dir) if args.output_dir else None
            print(f"\nGerando visualizacoes...")
            print(f"  Output dir: {output_dir}")
            print(f"  Sessao exemplo: {first_session_items}")
            create_visualizations(
                explorer,
                first_session_items,
                args.top_k,
                output_dir
            )
        
        print_final_summary(results_df)
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = output_dir / 'analysis_results.csv'
            results_df.to_csv(results_path, index=False)
            print(f"\nResultados salvos em: {results_path}")
        
        print_section_header("ANALISE CONCLUIDA", char="=")
    
    except Exception as e:
        print(f"\nERRO: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
