"""
Script de teste para verificar se modelos salvos podem ser carregados
e usados para fazer predições.
"""

import torch
import sys
from pathlib import Path

# project_root deve ser o diretório fermi
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_checkpoint_structure(checkpoint_path):
    """Testa estrutura do checkpoint."""
    print("\n" + "="*80)
    print(f" TESTE 1: Estrutura do Checkpoint")
    print("="*80)
    print(f"Arquivo: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("\n✓ Checkpoint carregado")
        
        # Verificar chaves essenciais
        required_keys = ['config', 'state_dict']
        for key in required_keys:
            if key in checkpoint:
                print(f"✓ Chave '{key}' encontrada")
            else:
                print(f"✗ Chave '{key}' NÃO encontrada")
                return False
        
        # Informações do config
        config = checkpoint['config']
        print(f"\nModelo: {config['model']}")
        print(f"Dataset: {config['dataset']}")
        print(f"Device original: {config['device']}")
        
        # Tamanho do state_dict
        print(f"\nState dict: {len(checkpoint['state_dict'])} parâmetros")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(checkpoint_path):
    """Testa carregamento do modelo com ModelExplorer."""
    print("\n" + "="*80)
    print(f" TESTE 2: Carregamento com ModelExplorer")
    print("="*80)
    
    try:
        from src.exploration.model_explorer import ModelExplorer
        
        print("Criando ModelExplorer...")
        explorer = ModelExplorer(checkpoint_path)
        
        print(f"\n✓ Modelo carregado com sucesso!")
        print(f"Config disponível: {explorer.config is not None}")
        print(f"Dataset disponível: {explorer.dataset is not None}")
        print(f"Modelo disponível: {explorer.model is not None}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction(checkpoint_path):
    """Testa predição com o modelo."""
    print("\n" + "="*80)
    print(f" TESTE 3: Geração de Recomendações")
    print("="*80)
    
    try:
        from src.exploration.model_explorer import ModelExplorer
        
        print("Carregando modelo...")
        explorer = ModelExplorer(checkpoint_path)
        
        # Testar predição com sessão dummy
        # Nota: IDs devem existir no dataset
        session_items = [1, 2, 3]  # IDs de exemplo
        
        print(f"\nGerando recomendações para sessão: {session_items}")
        recommendations = explorer.recommend_for_session(session_items, top_k=5)
        
        print(f"\n✓ Recomendações geradas!")
        print(f"Top-5: {recommendations}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro ao gerar predições: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes."""
    import glob
    
    print("="*80)
    print(" VERIFICAÇÃO DE MODELOS SALVOS")
    print("="*80)
    
    # Encontrar um checkpoint para testar
    checkpoint_dir = Path(project_root) / 'outputs' / 'saved'
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        print(f"\n✗ Nenhum checkpoint encontrado em {checkpoint_dir}")
        print("Execute alguns experimentos primeiro: make run-gru4rec")
        return
    
    # Testar o checkpoint mais recente
    checkpoint_path = str(sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1])
    
    print(f"\nTestando checkpoint mais recente:")
    print(f"{checkpoint_path}")
    
    # Executar testes
    test1 = test_checkpoint_structure(checkpoint_path)
    test2 = test_model_loading(checkpoint_path)
    test3 = test_prediction(checkpoint_path)
    
    # Resumo
    print("\n" + "="*80)
    print(" RESUMO DOS TESTES")
    print("="*80)
    
    results = {
        "Estrutura do checkpoint": test1,
        "Carregamento do modelo": test2,
        "Geração de recomendações": test3,
    }
    
    for test_name, passed in results.items():
        status = "✓ PASSOU" if passed else "✗ FALHOU"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print(" TODOS OS TESTES PASSARAM!")
        print(" Os modelos estão sendo salvos e carregados corretamente.")
    else:
        print(" ALGUNS TESTES FALHARAM!")
        print(" Verifique os erros acima.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
