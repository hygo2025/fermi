"""
Script de Exploração de Modelos

Exemplo de como usar o ModelExplorer para analisar recomendações
de um modelo treinado.

Uso:
    python src/exploration/explore_model.py --model outputs/saved/GRU4Rec-Dec-16-2024.pth \\
                                            --features /path/to/listings_features.parquet \\
                                            --session-ids 123,456,789
"""

import argparse
import sys
from pathlib import Path

# Adicionar root ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.exploration.model_explorer import ModelExplorer


def parse_args():
    parser = argparse.ArgumentParser(description='Explorar recomendações de modelo treinado')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Caminho para checkpoint do modelo (.pth)')
    
    parser.add_argument('--features', type=str, required=True,
                       help='Caminho para arquivo com features dos anúncios (CSV/Parquet)')
    
    parser.add_argument('--session-ids', type=str, required=True,
                       help='IDs dos anúncios da sessão separados por vírgula (ex: 123,456,789)')
    
    parser.add_argument('--top-k', type=int, default=10,
                       help='Número de recomendações a gerar (default: 10)')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Gerar visualização espacial (requer matplotlib)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print(" EXPLORADOR DE MODELOS - ANÁLISE DE RECOMENDAÇÕES")
    print("="*80)
    
    # Carregar modelo
    explorer = ModelExplorer(args.model)
    
    # Carregar features dos anúncios
    print(f"\nCarregando features de: {args.features}")
    explorer.load_item_features(args.features)
    
    # Parsear IDs da sessão
    session_ids = [int(x.strip()) for x in args.session_ids.split(',')]
    
    print(f"\nAnalisando sessão com {len(session_ids)} anúncios:")
    print(f"IDs: {session_ids}")
    
    # Gerar e analisar recomendações
    explorer.print_recommendation_report(session_ids, args.top_k)
    
    # Visualização espacial (opcional)
    if args.visualize:
        print("\nGerando visualização espacial...")
        explorer.visualize_spatial_distribution(session_ids, args.top_k)
    
    print("\n" + "="*80)
    print(" ANÁLISE CONCLUÍDA")
    print("="*80)


if __name__ == "__main__":
    main()
