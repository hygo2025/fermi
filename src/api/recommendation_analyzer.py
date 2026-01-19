"""
Analisador de Recommendations
Analisa espacialmente e semanticamente as recommendations do modelo
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
from recbole.quick_start import load_data_and_model


class RecommendationAnalyzer:
    """Analisa recommendations considerando localização e características dos imóveis"""
    
    def __init__(self, 
                 model_path: str,
                 listings_path: str,
                 data_path: str = 'data/recbole',
                 dataset_name: str = 'kepler'):
        """
        Args:
            model_path: Caminho para o modelo treinado (.pth)
            listings_path: Caminho para o parquet de listings
            data_path: Caminho para dados RecBole
            dataset_name: Nome do dataset
        """
        self.model_path = Path(model_path)
        self.listings_path = listings_path
        self.data_path = data_path
        self.dataset_name = dataset_name
        
        print("Loading data...")
        self._load_listings()
        self._load_model()
        
    def _load_listings(self):
        """Carrega dados dos listings"""
        print(f"  Loading listings from {self.listings_path}...")
        self.listings_df = pd.read_parquet(self.listings_path)
        
        # Converter coordenadas para float
        self.listings_df['lat'] = pd.to_numeric(self.listings_df['lat_region'], errors='coerce')
        self.listings_df['lon'] = pd.to_numeric(self.listings_df['lon_region'], errors='coerce')
        
        # Criar dicionário item_id -> listing usando listing_id_numeric
        self.item_to_listing = {}
        if 'listing_id_numeric' in self.listings_df.columns:
            for _, row in self.listings_df.iterrows():
                item_id = int(row['listing_id_numeric'])
                self.item_to_listing[item_id] = row.to_dict()
        
        print(f"  Success: {len(self.listings_df):,} listings loaded")
        print(f"  Mapped {len(self.item_to_listing):,} items")
        
    def _load_model(self):
        """Carrega modelo RecBole"""
        print(f"  Loading model from {self.model_path}...")
        
        # Load model without external config file
        # RecBole will infer config from the model checkpoint
        config_dict = {
            'data_path': self.data_path,
            'checkpoint_dir': str(self.model_path.parent)
        }
        
        # Corrigir compatibilidade PyTorch 2.6
        import torch
        torch.serialization.add_safe_globals([set])
        
        # Carregar modelo com weights_only=False (modelo é de fonte confiável)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Monkey patch para torch.load
            original_load = torch.load
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = safe_load
            
            try:
                self.config, self.model, self.dataset, _, _, _ = load_data_and_model(
                    model_file=str(self.model_path)
                )
            finally:
                torch.load = original_load
        
        print(f"  Model loaded successfully")
        
    def get_recommendations(self, session_items: List[int], top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Obtém recommendations para uma sessão
        
        Args:
            session_items: Lista de item_ids originais da sessão
            top_k: Número de recommendations
            
        Returns:
            Lista de (item_id_original, score) tuplas
        """
        import torch
        from recbole.data.interaction import Interaction
        
        # Clear CUDA cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Limit session items to avoid memory issues
        if len(session_items) > 20:
            session_items = session_items[-20:]  # Use last 20 items
        
        # Map original item IDs to RecBole internal IDs
        mapped_session_items = []
        item_field = self.config['ITEM_ID_FIELD']
        
        for orig_id in session_items:
            try:
                # RecBole usa token2id para mapear
                internal_id = self.dataset.token2id(item_field, str(orig_id))
                if internal_id is not None and internal_id != 0:  # 0 é padding
                    mapped_session_items.append(internal_id)
            except:
                # Item não existe no vocabulário do modelo
                continue
        
        if not mapped_session_items:
            print(f"⚠️  No valid items in session. Original IDs: {session_items}")
            return []
        
        # Obter device do modelo
        device = next(self.model.parameters()).device
        
        try:
            # Preparar input com IDs mapeados
            user_id = 0  # Dummy user
            item_seq = torch.tensor([mapped_session_items], dtype=torch.long).to(device)
            item_seq_len = torch.tensor([len(mapped_session_items)], dtype=torch.long).to(device)
            
            # Criar interaction
            interaction = Interaction({
                self.config['USER_ID_FIELD']: torch.tensor([user_id]).to(device),
                item_field + self.config['LIST_SUFFIX']: item_seq,
                self.config['ITEM_LIST_LENGTH_FIELD']: item_seq_len
            })
            
            # Fazer predição
            self.model.eval()
            with torch.no_grad():
                scores = self.model.full_sort_predict(interaction)
                scores = scores.view(-1)
                
                # Remover itens já vistos (usando IDs internos)
                for item in mapped_session_items:
                    if 0 <= item < len(scores):
                        scores[item] = -float('inf')
                
                # Top-K
                topk_scores, topk_items = torch.topk(scores, min(top_k, len(scores)))
            
            # Converter IDs internos de volta para IDs originais
            recommendations = []
            for internal_id, score in zip(topk_items.cpu().numpy(), topk_scores.cpu().numpy()):
                internal_id = int(internal_id)
                # Map back to original ID
                try:
                    orig_id = int(self.dataset.id2token(item_field, internal_id))
                    if orig_id in self.item_to_listing:
                        recommendations.append((orig_id, float(score)))
                except:
                    continue
            
        except RuntimeError as e:
            # If CUDA error, retry on CPU
            if 'CUDA' in str(e) or 'cuDNN' in str(e) or 'assert' in str(e).lower():
                print(f"⚠️  CUDA error, retrying on CPU: {e}")
                
                # Reset CUDA
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # Move model to CPU
                self.model = self.model.cpu()
                device = torch.device('cpu')
                
                # Preparar input
                user_id = 0
                item_seq = torch.tensor([mapped_session_items], dtype=torch.long).to(device)
                item_seq_len = torch.tensor([len(mapped_session_items)], dtype=torch.long).to(device)
                
                # Criar interaction
                interaction = Interaction({
                    self.config['USER_ID_FIELD']: torch.tensor([user_id]).to(device),
                    item_field + self.config['LIST_SUFFIX']: item_seq,
                    self.config['ITEM_LIST_LENGTH_FIELD']: item_seq_len
                })
                
                # Fazer predição
                self.model.eval()
                with torch.no_grad():
                    scores = self.model.full_sort_predict(interaction)
                    scores = scores.view(-1)
                    
                    # Remover itens já vistos
                    for item in mapped_session_items:
                        if 0 <= item < len(scores):
                            scores[item] = -float('inf')
                    
                    # Top-K
                    topk_scores, topk_items = torch.topk(scores, min(top_k, len(scores)))
                
                # Converter IDs internos de volta para IDs originais
                recommendations = []
                for internal_id, score in zip(topk_items.cpu().numpy(), topk_scores.cpu().numpy()):
                    internal_id = int(internal_id)
                    try:
                        orig_id = int(self.dataset.id2token(item_field, internal_id))
                        if orig_id in self.item_to_listing:
                            recommendations.append((orig_id, float(score)))
                    except:
                        continue
                
                # Move back to GPU for next time
                if torch.cuda.is_available():
                    try:
                        self.model = self.model.cuda()
                    except:
                        pass  # Stay on CPU if GPU is broken
            else:
                raise
        
        finally:
            # Clear cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return recommendations
    
    def calculate_distance(self, item1: int, item2: int) -> float:
        """
        Calcula distância geodésica entre dois imóveis (em km)
        
        Args:
            item1, item2: IDs dos imóveis
            
        Returns:
            Distância em km (ou None se coordenadas não disponíveis)
        """
        listing1 = self.item_to_listing.get(item1)
        listing2 = self.item_to_listing.get(item2)
        
        if listing1 is None or listing2 is None:
            return None
            
        lat1, lon1 = listing1.get('lat'), listing1.get('lon')
        lat2, lon2 = listing2.get('lat'), listing2.get('lon')
        
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return None
            
        try:
            return geodesic((lat1, lon1), (lat2, lon2)).km
        except:
            return None
    
    def compare_features(self, item1: int, item2: int) -> Dict[str, any]:
        """
        Compara características de dois imóveis
        
        Args:
            item1, item2: IDs dos imóveis
            
        Returns:
            Dicionário com comparações
        """
        listing1 = self.item_to_listing.get(item1)
        listing2 = self.item_to_listing.get(item2)
        
        if listing1 is None or listing2 is None:
            return {}
        
        comparison = {}
        
        # Características numéricas
        numeric_fields = ['price', 'usable_areas', 'bedrooms', 'bathrooms', 
                         'parking_spaces', 'suites']
        
        for field in numeric_fields:
            val1 = listing1.get(field)
            val2 = listing2.get(field)
            
            if pd.notna(val1) and pd.notna(val2):
                diff = val2 - val1
                pct_diff = (diff / val1 * 100) if val1 != 0 else 0
                comparison[field] = {
                    'item1': val1,
                    'item2': val2,
                    'diff': diff,
                    'pct_diff': pct_diff
                }
        
        # Características categóricas
        categorical_fields = ['city', 'neighborhood', 'unit_type', 'state']
        
        for field in categorical_fields:
            val1 = listing1.get(field)
            val2 = listing2.get(field)
            comparison[field] = {
                'item1': val1,
                'item2': val2,
                'match': val1 == val2
            }
        
        return comparison
    
    def analyze_session(self, session_items: List[int], top_k: int = 10):
        """
        Analysis completa de uma sessão
        
        Args:
            session_items: Lista de item_ids da sessão
            top_k: Número de recommendations
            
        Returns:
            DataFrame com análise completa
        """
        print(f"\n{'='*80}")
        print(f"ANÁLISE DE RECOMENDAÇÕES")
        print(f"{'='*80}")
        print(f"\nSession: {session_items}")
        print(f"  {len(session_items)} itens viewed")
        
        # Obter recommendations
        print(f"\nGerando top-{top_k} recommendations...")
        recommendations = self.get_recommendations(session_items, top_k)
        
        # Analisar cada recomendação
        results = []
        
        for rank, (rec_item, score) in enumerate(recommendations, 1):
            print(f"\n[{rank}] Item {rec_item} (score: {score:.4f})")
            
            # Calcular distâncias para cada item da sessão
            distances = []
            for sess_item in session_items:
                dist = self.calculate_distance(sess_item, rec_item)
                if dist is not None:
                    distances.append(dist)
            
            avg_distance = np.mean(distances) if distances else None
            min_distance = np.min(distances) if distances else None
            
            if avg_distance:
                print(f"  Distância média: {avg_distance:.2f} km")
                print(f"  Distância mínima: {min_distance:.2f} km")
            
            # Comparar características com item mais próximo
            if distances:
                closest_sess_item = session_items[np.argmin(distances)]
                comparison = self.compare_features(closest_sess_item, rec_item)
                
                if 'price' in comparison:
                    print(f"  Preço: R$ {comparison['price']['item2']:,.2f} "
                          f"({comparison['price']['pct_diff']:+.1f}%)")
                
                if 'bedrooms' in comparison:
                    print(f"  Quartos: {comparison['bedrooms']['item2']}")
                
                if 'city' in comparison:
                    match_str = "Success:" if comparison['city']['match'] else "Failed:"
                    print(f"  Cidade: {comparison['city']['item2']} {match_str}")
            
            results.append({
                'rank': rank,
                'item_id': rec_item,
                'score': score,
                'avg_distance_km': avg_distance,
                'min_distance_km': min_distance
            })
        
        return pd.DataFrame(results)
    
    def plot_map(self, session_items: List[int], recommendations: List[Tuple[int, float]],
                 save_path: str = None, figsize=(15, 10)):
        """
        Plota mapa com sessão e recommendations
        
        Args:
            session_items: Itens da sessão
            recommendations: Lista de (item_id, score)
            save_path: Caminho para salvar figura
            figsize: Tamanho da figura
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Coletar coordenadas
        session_coords = []
        session_cities = []
        
        for item in session_items:
            listing = self.item_to_listing.get(item)
            if listing is not None:
                lat, lon = listing.get('lat'), listing.get('lon')
                if pd.notna(lat) and pd.notna(lon):
                    session_coords.append((lat, lon))
                    session_cities.append(listing.get('city', 'N/A'))
        
        rec_coords = []
        rec_scores = []
        rec_cities = []
        
        for item_id, score in recommendations:
            listing = self.item_to_listing.get(item_id)
            if listing is not None:
                lat, lon = listing.get('lat'), listing.get('lon')
                if pd.notna(lat) and pd.notna(lon):
                    rec_coords.append((lat, lon))
                    rec_scores.append(score)
                    rec_cities.append(listing.get('city', 'N/A'))
        
        # Plotar sessão (azul)
        if session_coords:
            session_lats, session_lons = zip(*session_coords)
            ax.scatter(session_lons, session_lats, 
                      c='blue', s=200, alpha=0.6, 
                      marker='o', edgecolors='darkblue', linewidth=2,
                      label=f'Session ({len(session_coords)} itens)', zorder=3)
            
            # Anotar cidades
            for i, (lat, lon) in enumerate(session_coords):
                ax.annotate(f"{session_cities[i][:10]}", 
                           (lon, lat), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # Plotar recommendations (vermelho com intensidade baseada no score)
        if rec_coords:
            rec_lats, rec_lons = zip(*rec_coords)
            
            # Normalizar scores para tamanho dos pontos
            sizes = [100 + 500 * (s / max(rec_scores)) for s in rec_scores]
            
            scatter = ax.scatter(rec_lons, rec_lats, 
                               c=rec_scores, cmap='Reds', 
                               s=sizes, alpha=0.6,
                               marker='^', edgecolors='darkred', linewidth=1,
                               label=f'Recommendations ({len(rec_coords)} itens)', zorder=2)
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Score da Recomendação', rotation=270, labelpad=20)
            
            # Anotar top-5
            for i, (lat, lon) in enumerate(rec_coords[:5]):
                ax.annotate(f"#{i+1} {rec_cities[i][:10]}", 
                           (lon, lat), 
                           xytext=(5, -15), textcoords='offset points',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='yellow', alpha=0.5))
        
        # Desenhar linhas conectando sessão às recommendations mais próximas
        if session_coords and rec_coords:
            for sess_lat, sess_lon in session_coords:
                # Encontrar recomendação mais próxima
                min_dist = float('inf')
                closest_rec = None
                
                for rec_lat, rec_lon in rec_coords:
                    dist = np.sqrt((sess_lat - rec_lat)**2 + (sess_lon - rec_lon)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_rec = (rec_lat, rec_lon)
                
                if closest_rec:
                    ax.plot([sess_lon, closest_rec[1]], 
                           [sess_lat, closest_rec[0]],
                           'k--', alpha=0.2, linewidth=0.5, zorder=1)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Spatial Analysis: Session vs Recommendations', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSuccess: Static map saved to: {save_path}")
        
        plt.show()
    
    def plot_interactive_map(self, session_items: List[int], recommendations: List[Tuple[int, float]],
                            save_path: str = None):
        """
        Cria mapa interativo com Folium (OpenStreetMap)
        
        Args:
            session_items: Itens da sessão
            recommendations: Lista de (item_id, score)
            save_path: Caminho para salvar HTML (default: analysis_output/interactive_map.html)
        """
        import folium
        from folium import plugins
        
        # Coletar coordenadas e informações
        session_data = []
        rec_data = []
        
        for item_id in session_items:
            listing = self.item_to_listing.get(item_id)
            if listing is not None:
                lat, lon = listing.get('lat'), listing.get('lon')
                if pd.notna(lat) and pd.notna(lon):
                    session_data.append({
                        'item_id': item_id,
                        'lat': lat,
                        'lon': lon,
                        'city': listing.get('city', 'N/A'),
                        'neighborhood': listing.get('neighborhood', 'N/A'),
                        'price': listing.get('price', 0),
                        'bedrooms': listing.get('bedrooms', 0),
                        'bathrooms': listing.get('bathrooms', 0),
                        'area': listing.get('usable_areas', 0),
                        'type': listing.get('unit_type', 'N/A')
                    })
        
        for rank, (item_id, score) in enumerate(recommendations, 1):
            listing = self.item_to_listing.get(item_id)
            if listing is not None:
                lat, lon = listing.get('lat'), listing.get('lon')
                if pd.notna(lat) and pd.notna(lon):
                    rec_data.append({
                        'rank': rank,
                        'item_id': item_id,
                        'score': score,
                        'lat': lat,
                        'lon': lon,
                        'city': listing.get('city', 'N/A'),
                        'neighborhood': listing.get('neighborhood', 'N/A'),
                        'price': listing.get('price', 0),
                        'bedrooms': listing.get('bedrooms', 0),
                        'bathrooms': listing.get('bathrooms', 0),
                        'area': listing.get('usable_areas', 0),
                        'type': listing.get('unit_type', 'N/A')
                    })
        
        if not session_data and not rec_data:
            print("⚠️ Nenhuma coordenada disponível para plotar")
            return
        
        # Calcular centro do mapa
        all_lats = [d['lat'] for d in session_data + rec_data]
        all_lons = [d['lon'] for d in session_data + rec_data]
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        # Criar mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Adicionar camada de satélite como opção
        folium.TileLayer('Esri.WorldImagery', name='Satélite').add_to(m)
        
        # Adicionar marcadores da sessão (azul)
        session_group = folium.FeatureGroup(name='Session Viewed', show=True)
        
        for data in session_data:
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4 style="color: #1f77b4; margin-bottom: 10px;">Location: SESSION</h4>
                <b>ID:</b> {data['item_id']}<br>
                <b>Cidade:</b> {data['city']}<br>
                <b>Bairro:</b> {data['neighborhood']}<br>
                <b>Preço:</b> R$ {data['price']:,.2f}<br>
                <b>Quartos:</b> {data['bedrooms']}<br>
                <b>Banheiros:</b> {data['bathrooms']}<br>
                <b>Área:</b> {data['area']:.0f} m²<br>
                <b>Tipo:</b> {data['type']}
            </div>
            """
            
            folium.CircleMarker(
                location=[data['lat'], data['lon']],
                radius=10,
                popup=folium.Popup(popup_html, max_width=300),
                color='darkblue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.7,
                weight=3
            ).add_to(session_group)
            
            # Label com cidade
            folium.Marker(
                location=[data['lat'], data['lon']],
                icon=folium.DivIcon(html=f"""
                    <div style="font-size: 10px; color: blue; font-weight: bold; 
                                text-shadow: 1px 1px 2px white;">
                        {data['city'][:10]}
                    </div>
                """)
            ).add_to(session_group)
        
        session_group.add_to(m)
        
        # Adicionar marcadores das recommendations (vermelho)
        rec_group = folium.FeatureGroup(name='🔺 Recommendations', show=True)
        
        # Normalizar scores para cores
        if rec_data:
            max_score = max(d['score'] for d in rec_data)
            min_score = min(d['score'] for d in rec_data)
            
            for data in rec_data:
                # Calcular distância mínima da sessão
                min_distance = float('inf')
                for sess in session_data:
                    from geopy.distance import geodesic
                    dist = geodesic((sess['lat'], sess['lon']), 
                                   (data['lat'], data['lon'])).km
                    min_distance = min(min_distance, dist)
                
                # Cor baseada no rank
                if data['rank'] <= 3:
                    color = 'red'
                    icon = '⭐'
                elif data['rank'] <= 5:
                    color = 'orange'
                    icon = '🔥'
                else:
                    color = 'lightred'
                    icon = 'Location:'
                
                popup_html = f"""
                <div style="font-family: Arial; width: 280px;">
                    <h4 style="color: #d62728; margin-bottom: 10px;">
                        {icon} RECOMENDAÇÃO #{data['rank']}
                    </h4>
                    <b>Score:</b> {data['score']:.2f}<br>
                    <b>Distância:</b> {min_distance:.2f} km<br>
                    <hr style="margin: 8px 0;">
                    <b>ID:</b> {data['item_id']}<br>
                    <b>Cidade:</b> {data['city']}<br>
                    <b>Bairro:</b> {data['neighborhood']}<br>
                    <b>Preço:</b> R$ {data['price']:,.2f}<br>
                    <b>Quartos:</b> {data['bedrooms']}<br>
                    <b>Banheiros:</b> {data['bathrooms']}<br>
                    <b>Área:</b> {data['area']:.0f} m²<br>
                    <b>Tipo:</b> {data['type']}
                </div>
                """
                
                # Tamanho baseado no rank
                radius = 15 - (data['rank'] - 1) * 0.5
                
                folium.CircleMarker(
                    location=[data['lat'], data['lon']],
                    radius=radius,
                    popup=folium.Popup(popup_html, max_width=350),
                    color='darkred',
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(rec_group)
                
                # Label com rank para top-5
                if data['rank'] <= 5:
                    folium.Marker(
                        location=[data['lat'], data['lon']],
                        icon=folium.DivIcon(html=f"""
                            <div style="font-size: 12px; color: red; font-weight: bold; 
                                        background: yellow; padding: 2px 5px; 
                                        border-radius: 3px; border: 1px solid red;
                                        text-shadow: none;">
                                #{data['rank']}
                            </div>
                        """)
                    ).add_to(rec_group)
        
        rec_group.add_to(m)
        
        # Adicionar linhas conectando sessão às recommendations próximas
        lines_group = folium.FeatureGroup(name='Conexões', show=False)
        
        for sess in session_data:
            # Encontrar 3 recommendations mais próximas
            distances = []
            for rec in rec_data[:10]:  # Apenas top-10
                from geopy.distance import geodesic
                dist = geodesic((sess['lat'], sess['lon']), 
                               (rec['lat'], rec['lon'])).km
                distances.append((dist, rec))
            
            # Ordenar e pegar top-3 mais próximas
            distances.sort(key=lambda x: x[0])
            for dist, rec in distances[:3]:
                folium.PolyLine(
                    locations=[[sess['lat'], sess['lon']], 
                              [rec['lat'], rec['lon']]],
                    color='gray',
                    weight=1,
                    opacity=0.4,
                    popup=f"Distância: {dist:.2f} km"
                ).add_to(lines_group)
        
        lines_group.add_to(m)
        
        # Adicionar controle de camadas
        folium.LayerControl(position='topright').add_to(m)
        
        # Adicionar mini-mapa
        plugins.MiniMap(toggle_display=True).add_to(m)
        
        # Adicionar medidor de distância
        plugins.MeasureControl(position='topleft').add_to(m)
        
        # Adicionar fullscreen
        plugins.Fullscreen(position='topleft').add_to(m)
        
        # Adicionar legenda
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: auto; 
                    background-color: white; z-index:9999; font-size:12px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
            <p style="margin: 5px 0;"><b>Legenda</b></p>
            <p style="margin: 5px 0;">Session viewed</p>
            <p style="margin: 5px 0;">Top 3 recommendations</p>
            <p style="margin: 5px 0;">🟠 Top 4-5</p>
            <p style="margin: 5px 0;">🟡 Demais recommendations</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Salvar
        if save_path is None:
            save_path = 'analysis_output/interactive_map.html'
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(save_path)
        
        print(f"\nSuccess: Interactive map saved to: {save_path}")
        print(f"  Abra no navegador para visualizar!")
        
        return m
    
    def generate_map_html(self, session_items: List[int], recommendations: List[Tuple[int, float]]) -> str:
        """
        Gera HTML do mapa para embedding em página web
        
        Args:
            session_items: Itens da sessão
            recommendations: Lista de (item_id, score)
            
        Returns:
            HTML string do mapa
        """
        import folium
        from folium import plugins
        
        # Coletar dados (mesmo código do plot_interactive_map)
        session_data = []
        rec_data = []
        
        for item_id in session_items:
            listing = self.item_to_listing.get(item_id)
            if listing is not None:
                lat, lon = listing.get('lat'), listing.get('lon')
                if pd.notna(lat) and pd.notna(lon):
                    session_data.append({
                        'item_id': item_id,
                        'lat': lat,
                        'lon': lon,
                        'city': listing.get('city', 'N/A'),
                        'neighborhood': listing.get('neighborhood', 'N/A'),
                        'price': listing.get('price', 0),
                        'bedrooms': listing.get('bedrooms', 0),
                        'bathrooms': listing.get('bathrooms', 0),
                        'area': listing.get('usable_areas', 0),
                        'type': listing.get('unit_type', 'N/A')
                    })
        
        for rank, (item_id, score) in enumerate(recommendations, 1):
            listing = self.item_to_listing.get(item_id)
            if listing is not None:
                lat, lon = listing.get('lat'), listing.get('lon')
                if pd.notna(lat) and pd.notna(lon):
                    rec_data.append({
                        'rank': rank,
                        'item_id': item_id,
                        'score': score,
                        'lat': lat,
                        'lon': lon,
                        'city': listing.get('city', 'N/A'),
                        'neighborhood': listing.get('neighborhood', 'N/A'),
                        'price': listing.get('price', 0),
                        'bedrooms': listing.get('bedrooms', 0),
                        'bathrooms': listing.get('bathrooms', 0),
                        'area': listing.get('usable_areas', 0),
                        'type': listing.get('unit_type', 'N/A')
                    })
        
        if not session_data and not rec_data:
            return "<div>Nenhuma coordenada disponível para plotar</div>"
        
        # Calcular centro
        all_lats = [d['lat'] for d in session_data + rec_data]
        all_lons = [d['lon'] for d in session_data + rec_data]
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        # Criar mapa - mesma configuração da página de sessão
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11
        )
        
        # Session (azul) - circle markers com número da posição
        for idx, data in enumerate(session_data, 1):
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4 style="color: #1f77b4;">Session Item #{idx}</h4>
                <b>ID:</b> {data['item_id']}<br>
                <b>City:</b> {data['city']}<br>
                <b>Neighborhood:</b> {data['neighborhood']}<br>
                <b>Price:</b> R$ {data['price']:,.0f}<br>
                <b>Bedrooms:</b> {data['bedrooms']:.0f}<br>
                <b>Bathrooms:</b> {data['bathrooms']:.0f}<br>
                <b>Area:</b> {data['area']:.0f} m²<br>
            </div>
            """
            
            layer = folium.FeatureGroup(
                name=f"Sessão #{idx} - ID {data['item_id']}",
                show=True
            )
            layer.add_to(m)
            
            # Circle marker azul com número
            folium.CircleMarker(
                location=[data['lat'], data['lon']],
                radius=12,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Position {idx}: {data['city']}",
                color='#2E86C1',
                fill=True,
                fillColor='#3498DB',
                fillOpacity=0.8,
                weight=2
            ).add_to(layer)
            
            # Adicionar número da posição no centro do círculo
            folium.Marker(
                location=[data['lat'], data['lon']],
                icon=folium.DivIcon(html=f"""
                    <div style="
                        font-size: 12px; 
                        color: white; 
                        font-weight: bold;
                        text-align: center;
                        margin-left: 0px;
                        margin-top: -3px;
                    ">{idx}</div>
                """)
            ).add_to(layer)
        
        # Recommendations (vermelho) - marcadores melhores
        if rec_data:
            for data in rec_data:
                # Calcular distância
                min_distance = float('inf')
                for sess in session_data:
                    from geopy.distance import geodesic
                    dist = geodesic((sess['lat'], sess['lon']), 
                                   (data['lat'], data['lon'])).km
                    min_distance = min(min_distance, dist)
                
                popup_html = f"""
                <div style="font-family: Arial; width: 280px;">
                    <h4 style="color: #d62728;">Recommendation #{data['rank']}</h4>
                    <b>Score:</b> {data['score']:.4f}<br>
                    <b>Distance:</b> {min_distance:.2f} km<br>
                    <hr>
                    <b>ID:</b> {data['item_id']}<br>
                    <b>City:</b> {data['city']}<br>
                    <b>Neighborhood:</b> {data['neighborhood']}<br>
                    <b>Price:</b> R$ {data['price']:,.0f}<br>
                    <b>Bedrooms:</b> {data['bedrooms']:.0f}<br>
                    <b>Bathrooms:</b> {data['bathrooms']:.0f}<br>
                    <b>Area:</b> {data['area']:.0f} m²<br>
                </div>
                """
                
                # Cor baseada no rank
                if data['rank'] <= 3:
                    color = 'green'
                    fill_color = '#2ECC71'
                    border_color = '#27AE60'
                elif data['rank'] <= 6:
                    color = 'orange'
                    fill_color = '#F39C12'
                    border_color = '#E67E22'
                else:
                    color = 'red'
                    fill_color = '#E74C3C'
                    border_color = '#C0392B'
                
                layer = folium.FeatureGroup(
                    name=f"Recomendação #{data['rank']} - ID {data['item_id']}",
                    show=True
                )
                layer.add_to(m)
                
                # Circle marker com número do rank
                folium.CircleMarker(
                    location=[data['lat'], data['lon']],
                    radius=12,
                    popup=folium.Popup(popup_html, max_width=350),
                    tooltip=f"Recommendation #{data['rank']}: {data['city']} (Score: {data['score']:.2f})",
                    color=border_color,
                    fill=True,
                    fillColor=fill_color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(layer)
                
                # Adicionar número do rank
                folium.Marker(
                    location=[data['lat'], data['lon']],
                    icon=folium.DivIcon(html=f"""
                        <div style="
                            font-size: 12px; 
                            color: white; 
                            font-weight: bold;
                            text-align: center;
                            margin-left: 0px;
                            margin-top: -3px;
                        ">{data['rank']}</div>
                    """)
                ).add_to(layer)
        
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Retornar HTML
        return m._repr_html_()
    
    def compare_session_vs_recommendations(self, session_items: List[int], 
                                          recommendations: List[Tuple[int, float]]) -> pd.DataFrame:
        """
        Compara características agregadas da sessão vs recommendations
        
        Args:
            session_items: Itens da sessão
            recommendations: Lista de (item_id, score)
            
        Returns:
            DataFrame com comparação
        """
        # Coletar características
        def get_stats(items):
            stats = {
                'price': [],
                'bedrooms': [],
                'bathrooms': [],
                'parking_spaces': [],
                'usable_areas': [],
                'cities': []
            }
            
            for item in items:
                listing = self.item_to_listing.get(item)
                if listing is not None:
                    for key in stats.keys():
                        if key == 'cities':
                            city = listing.get('city')
                            if pd.notna(city):
                                stats[key].append(city)
                        else:
                            val = listing.get(key)
                            if pd.notna(val):
                                stats[key].append(val)
            
            return stats
        
        session_stats = get_stats(session_items)
        rec_items = [item for item, _ in recommendations]
        rec_stats = get_stats(rec_items)
        
        # Criar comparação
        comparison = []
        
        for feature in ['price', 'bedrooms', 'bathrooms', 'parking_spaces', 'usable_areas']:
            if session_stats[feature] and rec_stats[feature]:
                comparison.append({
                    'Feature': feature,
                    'Session_Mean': np.mean(session_stats[feature]),
                    'Session_Std': np.std(session_stats[feature]),
                    'Rec_Mean': np.mean(rec_stats[feature]),
                    'Rec_Std': np.std(rec_stats[feature]),
                    'Diff_%': ((np.mean(rec_stats[feature]) - np.mean(session_stats[feature])) / 
                              np.mean(session_stats[feature]) * 100)
                })
        
        # Cidades mais frequentes
        if session_stats['cities'] and rec_stats['cities']:
            from collections import Counter
            
            session_cities = Counter(session_stats['cities'])
            rec_cities = Counter(rec_stats['cities'])
            
            comparison.append({
                'Feature': 'Top_City',
                'Session_Mean': session_cities.most_common(1)[0][0] if session_cities else 'N/A',
                'Session_Std': session_cities.most_common(1)[0][1] if session_cities else 0,
                'Rec_Mean': rec_cities.most_common(1)[0][0] if rec_cities else 'N/A',
                'Rec_Std': rec_cities.most_common(1)[0][1] if rec_cities else 0,
                'Diff_%': 0
            })
        
        return pd.DataFrame(comparison)


def main():
    """Exemplo de uso"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analisar recommendations espacialmente')
    parser.add_argument('--model', required=True, help='Caminho para modelo .pth')
    parser.add_argument('--listings', required=True, help='Caminho para listings.parquet')
    parser.add_argument('--session', nargs='+', type=int, required=True,
                       help='IDs dos itens da sessão')
    parser.add_argument('--top-k', type=int, default=10, help='Número de recommendations')
    parser.add_argument('--output', default='analysis_output',
                       help='Diretório para outputs')
    
    args = parser.parse_args()
    
    # Criar diretório de output
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Inicializar analisador
    analyzer = RecommendationAnalyzer(
        model_path=args.model,
        listings_path=args.listings
    )
    
    # Analysis completa
    results_df = analyzer.analyze_session(args.session, top_k=args.top_k)
    results_df.to_csv(output_dir / 'recommendations_analysis.csv', index=False)
    
    # Obter recommendations
    recommendations = analyzer.get_recommendations(args.session, top_k=args.top_k)
    
    # Plotar mapa estático
    analyzer.plot_map(
        args.session, 
        recommendations,
        save_path=output_dir / 'spatial_map.png'
    )
    
    # Plotar mapa interativo
    analyzer.plot_interactive_map(
        args.session,
        recommendations,
        save_path=output_dir / 'interactive_map.html'
    )
    
    # Comparação de características
    comparison_df = analyzer.compare_session_vs_recommendations(args.session, recommendations)
    comparison_df.to_csv(output_dir / 'feature_comparison.csv', index=False)
    
    print(f"\n{'='*80}")
    print("RESUMO DA ANÁLISE")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    print(f"\nSuccess: Analysis saved to: {output_dir}")


if __name__ == '__main__':
    main()
