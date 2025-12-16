"""
Explorador de Modelos - Análise de Recomendações

Permite carregar modelos treinados e analisar as recomendações geradas,
incluindo características dos anúncios recomendados.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model


class ModelExplorer:
    """
    Explorador de modelos treinados para análise de recomendações.
    """
    
    def __init__(self, model_path: str, dataset_path: str = None):
        """
        Inicializa o explorador com um modelo treinado.
        
        Args:
            model_path: Caminho para o checkpoint do modelo (.pth)
            dataset_path: Caminho para os dados RecBole (opcional, se quiser recriar dataset)
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.config = None
        self.dataset = None
        self.item_features = None
        
        print(f"\nCarregando modelo de: {model_path}")
        self._load_model()
        
    def _load_model(self):
        """Carrega o modelo e configurações."""
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Extrair configuração do checkpoint
        self.config = checkpoint['config']
        
        # Tentar criar dataset (pode falhar se dados não existirem)
        try:
            # Se dataset_path foi fornecido, usar ele
            if self.dataset_path:
                self.config['data_path'] = self.dataset_path
            
            self.dataset = create_dataset(self.config)
            print(f"Dataset carregado: {self.config['dataset']}")
        except (FileNotFoundError, ValueError) as e:
            print(f"AVISO: Dataset original não encontrado: {e}")
            print("   Modo simplificado: use apenas IDs numéricos para predição")
            
            # Criar dataset fake mínimo para inicializar o modelo
            from recbole.data.dataset import Dataset
            
            # Extrair informações do checkpoint
            state_dict = checkpoint['state_dict']
            
            # Descobrir número de items pelo embedding
            for key in state_dict.keys():
                if 'item' in key.lower() and 'embedding' in key.lower():
                    n_items = state_dict[key].shape[0]
                    break
            else:
                n_items = 30000  # Fallback
            
            print(f"   Detectado: ~{n_items} items no modelo")
            
            # Criar dataset fake
            class FakeDataset:
                def __init__(self, n_items):
                    self.item_num = n_items
                    self.user_num = 1000
                    
            self.dataset = FakeDataset(n_items)
        
        # Criar modelo
        model_name = self.config['model']
        self.model = get_model(model_name)(self.config, self.dataset).to(self.config['device'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        print(f"OK - Modelo {model_name} carregado com sucesso!")
        print(f"  Items: {self.dataset.item_num}")
        print(f"  Device: {self.config['device']}")
        
    def load_item_features(self, features_path: str):
        """
        Carrega características dos anúncios para análise.
        
        Args:
            features_path: Caminho para CSV/Parquet com features dos anúncios
        """
        if features_path.endswith('.parquet'):
            self.item_features = pd.read_parquet(features_path)
        else:
            self.item_features = pd.read_csv(features_path)
        
        # Detectar coluna de ID (pode ser 'listing_id', 'listing_id_numeric', 'item_id', etc.)
        id_columns = ['listing_id', 'listing_id_numeric', 'item_id', 'id']
        self.item_id_col = None
        
        for col in id_columns:
            if col in self.item_features.columns:
                self.item_id_col = col
                break
        
        if self.item_id_col is None:
            raise ValueError(f"Nenhuma coluna de ID encontrada. Esperado uma de: {id_columns}")
        
        # Se não for 'listing_id', criar alias
        if self.item_id_col != 'listing_id':
            self.item_features['listing_id'] = self.item_features[self.item_id_col]
            
        print(f"Features carregadas: {len(self.item_features)} anúncios")
        print(f"Coluna de ID: {self.item_id_col}")
        print(f"Colunas disponíveis: {list(self.item_features.columns)}")
        
    def recommend_for_session(self, session_items: List[int], top_k: int = 10) -> List[int]:
        """
        Gera recomendações para uma sessão.
        
        Args:
            session_items: Lista de IDs de anúncios na sessão atual
            top_k: Número de recomendações a retornar
            
        Returns:
            Lista com top-K anúncios recomendados
        """
        with torch.no_grad():
            # Preparar dados da sessão
            max_len = self.config['MAX_ITEM_LIST_LENGTH']
            session_padded = session_items[-max_len:] if len(session_items) > max_len else session_items
            
            # Pad se necessário
            if len(session_padded) < max_len:
                padding = [0] * (max_len - len(session_padded))
                session_padded = padding + session_padded
            
            interaction_dict = {
                'item_id_list': torch.LongTensor([session_padded]),
                'item_length': torch.LongTensor([len(session_items)]),
                self.config['ITEM_ID_FIELD']: torch.LongTensor([session_items[-1]]),
                self.config['SESSION_ID_FIELD']: torch.LongTensor([0]),
            }
            
            # Converter para Interaction
            from recbole.data.interaction import Interaction
            interaction = Interaction(interaction_dict)
            interaction = interaction.to(self.config['device'])
            
            # Gerar scores
            scores = self.model.full_sort_predict(interaction)
            scores = scores.view(-1).cpu()
            
            # Excluir itens já vistos
            for item_id in session_items:
                if item_id < len(scores):
                    scores[item_id] = -np.inf
            
            # Top-K
            _, topk_indices = torch.topk(scores, min(top_k, len(scores)))
            
            return topk_indices.numpy().tolist()
    
    def analyze_recommendations(self, session_items: List[int], top_k: int = 10) -> pd.DataFrame:
        """
        Gera recomendações e retorna análise com features dos anúncios.
        
        Args:
            session_items: Lista de IDs na sessão
            top_k: Número de recomendações
            
        Returns:
            DataFrame com recomendações e suas características
        """
        if self.item_features is None:
            raise ValueError("Carregue as features primeiro com load_item_features()")
        
        # Gerar recomendações
        recommended_ids = self.recommend_for_session(session_items, top_k)
        
        # Buscar features dos recomendados
        rec_features = self.item_features[self.item_features['listing_id'].isin(recommended_ids)].copy()
        
        # Adicionar ranking
        rec_features['rank'] = rec_features['listing_id'].map(
            {item_id: rank + 1 for rank, item_id in enumerate(recommended_ids)}
        )
        rec_features = rec_features.sort_values('rank')
        
        return rec_features
    
    def compare_with_session(self, session_items: List[int], top_k: int = 10) -> Dict:
        """
        Compara características dos anúncios da sessão com os recomendados.
        
        Args:
            session_items: IDs da sessão atual
            top_k: Número de recomendações
            
        Returns:
            Dicionário com estatísticas comparativas
        """
        if self.item_features is None:
            raise ValueError("Carregue as features primeiro")
        
        # Features da sessão
        session_features = self.item_features[self.item_features['listing_id'].isin(session_items)]
        
        # Features das recomendações
        recommended_ids = self.recommend_for_session(session_items, top_k)
        rec_features = self.item_features[self.item_features['listing_id'].isin(recommended_ids)]
        
        comparison = {
            'session': self._compute_stats(session_features),
            'recommended': self._compute_stats(rec_features)
        }
        
        return comparison
    
    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Computa estatísticas das features."""
        stats = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'listing_id':
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std()
                }
        
        # Adicionar contagens de categorias
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['listing_id']:
                stats[f'{col}_top3'] = df[col].value_counts().head(3).to_dict()
        
        return stats
    
    def visualize_spatial_distribution(self, session_items: List[int], top_k: int = 10):
        """
        Visualiza distribuição espacial (se houver lat/lon).
        
        Args:
            session_items: IDs da sessão
            top_k: Número de recomendações
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não instalado. Instale com: pip install matplotlib")
            return
        
        if self.item_features is None or 'latitude' not in self.item_features.columns:
            print("Features geográficas não disponíveis")
            return
        
        # Dados da sessão
        session_df = self.item_features[self.item_features['listing_id'].isin(session_items)]
        
        # Recomendações
        rec_ids = self.recommend_for_session(session_items, top_k)
        rec_df = self.item_features[self.item_features['listing_id'].isin(rec_ids)]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(session_df['longitude'], session_df['latitude'], 
                   c='blue', s=100, alpha=0.6, label='Sessão', marker='o')
        plt.scatter(rec_df['longitude'], rec_df['latitude'], 
                   c='red', s=100, alpha=0.6, label='Recomendados', marker='^')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Distribuição Espacial: Sessão vs Recomendações')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def print_recommendation_report(self, session_items: List[int], top_k: int = 10):
        """
        Imprime relatório completo das recomendações.
        
        Args:
            session_items: IDs da sessão
            top_k: Número de recomendações
        """
        print("\n" + "="*80)
        print("RELATÓRIO DE RECOMENDAÇÕES")
        print("="*80)
        
        print(f"\nSessão atual: {len(session_items)} anúncios")
        
        # Recomendações
        rec_df = self.analyze_recommendations(session_items, top_k)
        
        print(f"\nTop-{top_k} Recomendações:")
        print("-" * 80)
        
        # Colunas relevantes para mostrar
        display_cols = ['rank', 'listing_id']
        optional_cols = ['price', 'city', 'neighborhood', 'bedrooms', 'bathrooms', 
                        'area', 'latitude', 'longitude']
        
        for col in optional_cols:
            if col in rec_df.columns:
                display_cols.append(col)
        
        print(rec_df[display_cols].to_string(index=False))
        
        # Comparação estatística
        print("\n" + "="*80)
        print("COMPARAÇÃO: SESSÃO vs RECOMENDAÇÕES")
        print("="*80)
        
        comparison = self.compare_with_session(session_items, top_k)
        
        print("\nSessão:")
        self._print_stats(comparison['session'])
        
        print("\nRecomendações:")
        self._print_stats(comparison['recommended'])
        
    def _print_stats(self, stats: Dict):
        """Imprime estatísticas formatadas."""
        for key, value in stats.items():
            if isinstance(value, dict) and 'mean' in value:
                print(f"  {key}: média={value['mean']:.2f}, "
                      f"mediana={value['median']:.2f}, std={value['std']:.2f}")
            elif isinstance(value, dict):
                print(f"  {key}: {value}")
