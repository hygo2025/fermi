#!/usr/bin/env python3
"""
Script para limpar checkpoints antigos e manter apenas os melhores modelos.

Estratégias:
1. keep-best-per-model: Mantém apenas o melhor checkpoint de cada modelo
2. keep-recent: Mantém apenas os N checkpoints mais recentes
3. clean-all: Remove todos os checkpoints
"""

import argparse
from pathlib import Path
from datetime import datetime
import re


def parse_checkpoint_name(filename):
    """
    Extrai informações do nome do checkpoint.
    Formato: MODEL-Mon-DD-YYYY_HH-MM-SS.pth
    """
    parts = filename.stem.split('-')
    if len(parts) < 2:
        return None, None
    
    model_name = parts[0]
    # Tentar extrair timestamp
    try:
        date_str = '-'.join(parts[1:])
        # Exemplo: Dec-16-2025_12-45-30
        timestamp = datetime.strptime(date_str, '%b-%d-%Y_%H-%M-%S')
        return model_name, timestamp
    except:
        return model_name, None


def get_checkpoint_score(checkpoint_path):
    """Extrai o score do checkpoint (best_valid_score)."""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint.get('best_valid_score', 0)
    except:
        return 0


def keep_best_per_model(checkpoint_dir):
    """Mantém apenas o melhor checkpoint de cada modelo."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        print("Nenhum checkpoint encontrado")
        return
    
    # Agrupar por modelo
    models = {}
    for ckpt in checkpoints:
        model_name, _ = parse_checkpoint_name(ckpt)
        if model_name:
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(ckpt)
    
    removed = 0
    kept = 0
    
    print(f"\nEncontrados {len(checkpoints)} checkpoints de {len(models)} modelos")
    print("\nAnalisando cada modelo...")
    
    for model_name, ckpts in models.items():
        print(f"\n{model_name}: {len(ckpts)} checkpoints")
        
        # Encontrar o melhor
        best_ckpt = None
        best_score = -float('inf')
        
        for ckpt in ckpts:
            score = get_checkpoint_score(ckpt)
            print(f"  {ckpt.name}: score={score:.4f}")
            if score > best_score:
                best_score = score
                best_ckpt = ckpt
        
        print(f"  → Melhor: {best_ckpt.name} (score={best_score:.4f})")
        
        # Remover os outros
        for ckpt in ckpts:
            if ckpt != best_ckpt:
                ckpt.unlink()
                removed += 1
                print(f"    ✗ Removido: {ckpt.name}")
            else:
                kept += 1
    
    print(f"\n{'='*60}")
    print(f"Resumo:")
    print(f"  Mantidos: {kept} checkpoints")
    print(f"  Removidos: {removed} checkpoints")
    print(f"  Espaço liberado: ~{removed * 200} MB (estimativa)")
    print(f"{'='*60}")


def keep_recent(checkpoint_dir, n=5):
    """Mantém apenas os N checkpoints mais recentes."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        print("Nenhum checkpoint encontrado")
        return
    
    # Ordenar por data de modificação (mais recente primeiro)
    checkpoints_sorted = sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
    
    to_keep = checkpoints_sorted[:n]
    to_remove = checkpoints_sorted[n:]
    
    print(f"\nEncontrados {len(checkpoints)} checkpoints")
    print(f"Mantendo os {n} mais recentes:")
    for ckpt in to_keep:
        print(f"  ✓ {ckpt.name}")
    
    if to_remove:
        print(f"\nRemovendo {len(to_remove)} checkpoints antigos:")
        for ckpt in to_remove:
            print(f"  ✗ {ckpt.name}")
            ckpt.unlink()
    
    print(f"\nEspaço liberado: ~{len(to_remove) * 200} MB (estimativa)")


def clean_all(checkpoint_dir):
    """Remove todos os checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        print("Nenhum checkpoint encontrado")
        return
    
    print(f"\n⚠️  ATENÇÃO: Removendo TODOS os {len(checkpoints)} checkpoints!")
    response = input("Tem certeza? (sim/não): ")
    
    if response.lower() == 'sim':
        for ckpt in checkpoints:
            ckpt.unlink()
            print(f"  ✗ Removido: {ckpt.name}")
        print(f"\n{len(checkpoints)} checkpoints removidos")
        print(f"Espaço liberado: ~{len(checkpoints) * 200} MB (estimativa)")
    else:
        print("Operação cancelada")


def show_stats(checkpoint_dir):
    """Mostra estatísticas dos checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        print("Nenhum checkpoint encontrado")
        return
    
    # Calcular estatísticas
    total_size = sum(ckpt.stat().st_size for ckpt in checkpoints)
    
    # Agrupar por modelo
    models = {}
    for ckpt in checkpoints:
        model_name, _ = parse_checkpoint_name(ckpt)
        if model_name:
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(ckpt)
    
    print(f"\n{'='*60}")
    print(" ESTATÍSTICAS DE CHECKPOINTS")
    print(f"{'='*60}")
    print(f"\nTotal: {len(checkpoints)} checkpoints")
    print(f"Tamanho total: {total_size / (1024**3):.2f} GB")
    print(f"Tamanho médio: {total_size / len(checkpoints) / (1024**2):.1f} MB")
    
    print(f"\nPor modelo:")
    for model_name, ckpts in sorted(models.items()):
        model_size = sum(ckpt.stat().st_size for ckpt in ckpts)
        print(f"  {model_name}: {len(ckpts)} checkpoints, {model_size / (1024**2):.1f} MB")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Gerenciar checkpoints de modelos')
    
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/saved',
                       help='Diretório com checkpoints')
    
    parser.add_argument('--strategy', type=str, choices=['stats', 'keep-best', 'keep-recent', 'clean-all'],
                       default='stats',
                       help='Estratégia de limpeza')
    
    parser.add_argument('--keep-n', type=int, default=5,
                       help='Número de checkpoints a manter (para keep-recent)')
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Diretório não encontrado: {checkpoint_dir}")
        return
    
    if args.strategy == 'stats':
        show_stats(checkpoint_dir)
    elif args.strategy == 'keep-best':
        keep_best_per_model(checkpoint_dir)
    elif args.strategy == 'keep-recent':
        keep_recent(checkpoint_dir, args.keep_n)
    elif args.strategy == 'clean-all':
        clean_all(checkpoint_dir)


if __name__ == "__main__":
    main()
