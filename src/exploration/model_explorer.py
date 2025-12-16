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
    """Explore trained recommendation models"""
    
    def __init__(self, model_path: str, dataset_path: str = None):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.config = None
        self.dataset = None
        self.item_features = None
        
        print(f"\nCarregando modelo de: {model_path}")
        self._load_model()
        
    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        self.config = checkpoint['config']
        
        try:
            if self.dataset_path:
                self.config['data_path'] = self.dataset_path
            
            from recbole.data import create_dataset
            self.dataset = create_dataset(self.config)
            print(f"Dataset carregado: {self.config['dataset']}")
            
            # Criar mapeamento token -> ID interno
            self.token2id = {}
            if hasattr(self.dataset, 'field2id_token'):
                # Tentar ambos os formatos de nome de campo
                item_field = 'item_id:token' if 'item_id:token' in self.dataset.field2id_token else 'item_id'
                
                if item_field in self.dataset.field2id_token:
                    tokens = self.dataset.field2id_token[item_field]
                    self.token2id = {str(token): idx for idx, token in enumerate(tokens)}
                    print(f"  Token mapping criado: {len(self.token2id)} items")
                    print(f"  Amostra: {list(self.token2id.items())[:5]}")
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
        Load item features for analysis.
        
        Args:
            features_path: Path to CSV/Parquet with item features
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
        
    def load_real_sessions(self, inter_file: str, n_sessions: int = 10) -> List[Tuple[int, List[int]]]:
        """
        Load real user sessions from interaction file.
        
        Args:
            inter_file: Path to .inter file with interactions
            n_sessions: Number of random sessions to load
            
        Returns:
            List of tuples (session_id, item_list)
        """
        df = pd.read_csv(inter_file, sep='\t')
        
        # Group by session
        session_col = self.config.get('SESSION_ID_FIELD', 'session_id')
        item_col = self.config.get('ITEM_ID_FIELD', 'item_id')
        time_col = self.config.get('TIME_FIELD', 'timestamp')
        
        # Sort by time within each session
        df = df.sort_values([session_col, time_col])
        
        # Get sessions with at least 3 items
        session_groups = df.groupby(session_col)[item_col].apply(list)
        valid_sessions = session_groups[session_groups.apply(len) >= 3]
        
        # Sample random sessions
        sampled = valid_sessions.sample(min(n_sessions, len(valid_sessions)))
        
        return [(sid, items) for sid, items in sampled.items()]
    
    def recommend_for_session(self, session_items: List[int], top_k: int = 10) -> List[int]:
        """
        Generate recommendations for a session.
        
        Args:
            session_items: List of item IDs (tokens originais ou IDs remapeados)
            top_k: Number of recommendations to return
            
        Returns:
            List with top-K recommended items (IDs remapeados)
        """
        with torch.no_grad():
            # Se temos mapeamento, converter tokens para IDs internos
            if hasattr(self, 'token2id') and self.token2id:
                mapped_items = []
                for item in session_items:
                    str_item = str(item)
                    if str_item in self.token2id:
                        mapped_items.append(self.token2id[str_item])
                    elif 0 <= item <= self.dataset.item_num - 1:
                        # Já é um ID interno válido
                        mapped_items.append(item)
                
                if len(mapped_items) == 0:
                    print(f"AVISO: Nenhum item valido apos mapeamento")
                    return []
                
                if len(mapped_items) < len(session_items):
                    unmapped = set(session_items) - set([int(k) for k in self.token2id.keys() if self.token2id[k] in mapped_items])
                    print(f"AVISO: {len(unmapped)} items nao encontrados no vocabulario")
                
                valid_items = mapped_items
            else:
                # Sem mapeamento, validar IDs diretamente
                max_item_id = self.dataset.item_num - 1
                valid_items = [item_id for item_id in session_items if 0 <= item_id <= max_item_id]
                
                if len(valid_items) == 0:
                    print(f"AVISO: Nenhum item valido na sessao (max_id={max_item_id})")
                    return []
                
                if len(valid_items) < len(session_items):
                    invalid = set(session_items) - set(valid_items)
                    print(f"AVISO: {len(invalid)} itens invalidos removidos")
            
            # Preparar dados da sessão
            max_len = self.config['MAX_ITEM_LIST_LENGTH']
            session_padded = valid_items[-max_len:] if len(valid_items) > max_len else valid_items
            
            # Pad se necessário
            if len(session_padded) < max_len:
                padding = [0] * (max_len - len(session_padded))
                session_padded = padding + session_padded
            
            interaction_dict = {
                'item_id_list': torch.LongTensor([session_padded]),
                'item_length': torch.LongTensor([len(valid_items)]),
                self.config['ITEM_ID_FIELD']: torch.LongTensor([valid_items[-1]]),
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
            for item_id in valid_items:
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
