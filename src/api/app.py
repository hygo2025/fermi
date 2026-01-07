"""
FastAPI Web Application for Real Estate Recommendations
"""
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import sys
import logging
from typing import List, Optional, Dict
import json
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.recommendation_analyzer import RecommendationAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Fermi Recommendation API",
    description="Real Estate Recommendation System with Spatial Analysis",
    version="1.0.0"
)

# Templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Global analyzer (loaded on startup)
analyzer: Optional[RecommendationAnalyzer] = None

# Global sessions data
sessions_df: Optional[pd.DataFrame] = None


@app.on_event("startup")
async def startup_event():
    """Initialize recommendation analyzer on startup"""
    global analyzer, sessions_df
    
    try:
        # Get model from command line arguments
        args = parse_args()
        
        # Check if it's a model name or path
        if args.model.endswith('.pth'):
            model_path = Path(args.model)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {args.model}")
            latest_model = model_path
        else:
            # Search for model by name in outputs/saved/
            models_dir = Path("outputs/saved")
            model_files = list(models_dir.glob(f"{args.model}-*.pth"))
            
            if not model_files:
                raise FileNotFoundError(f"No {args.model} model found in {models_dir}/")
            
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Loading model: {latest_model}")
        
        from src.utils.enviroment import get_config
        config = get_config()
        
        analyzer = RecommendationAnalyzer(
            model_path=str(latest_model),
            listings_path=config['raw_data']['listings_processed_path'],
            data_path=config['data_path'],
            dataset_name=config['dataset']
        )
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Total {len(analyzer.item_to_listing):,} listings available")
        
        # Load sessions data
        sessions_path = config['raw_data'].get('sessions_path', config['raw_data']['events_processed_path'])
        logger.info(f"Loading sessions from: {sessions_path}")
        sessions_df = pd.read_parquet(sessions_path)
        logger.info(f"Loaded {len(sessions_df):,} session events")
        logger.info(f"Total unique sessions: {sessions_df['session_id'].nunique():,}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page redirects to sessions list"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/sessions")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": analyzer is not None,
        "items_count": len(analyzer.item_to_listing) if analyzer else 0
    }


@app.get("/api/items")
async def get_items(limit: int = 100):
    """Get available item IDs"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    item_ids = list(analyzer.item_to_listing.keys())[:limit]
    return {"items": item_ids, "total": len(analyzer.item_to_listing)}


@app.get("/api/recommend")
async def api_recommend(session_ids: str, top_k: int = 10):
    """API endpoint for recommendations (JSON response)"""
    try:
        session_items = [int(x.strip()) for x in session_ids.split(",") if x.strip()]
        
        if not session_items:
            raise HTTPException(status_code=400, detail="No session IDs provided")
        
        recommendations = analyzer.get_recommendations(session_items, top_k=top_k)
        
        return {
            "session": session_items,
            "recommendations": [
                {"rank": i, "item_id": item_id, "score": float(score)}
                for i, (item_id, score) in enumerate(recommendations, 1)
            ]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_class=HTMLResponse)
async def list_sessions(
    request: Request, 
    page: int = 1, 
    per_page: int = 50,
    sort_by: str = "first_event",
    sort_order: str = "desc"
):
    """List all sessions with pagination and sorting"""
    if sessions_df is None or analyzer is None:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": "Sessions data not loaded"}
        )
    
    # Get unique sessions
    session_groups = sessions_df.groupby('session_id').agg({
        'user_id': 'first',
        'timestamp': ['min', 'max'],
        'item_id': 'count'
    }).reset_index()
    
    session_groups.columns = ['session_id', 'user_id', 'first_event', 'last_event', 'event_count']
    
    # Apply sorting
    valid_sort_columns = ['session_id', 'event_count', 'first_event', 'last_event']
    if sort_by not in valid_sort_columns:
        sort_by = 'first_event'
    
    ascending = (sort_order == 'asc')
    session_groups = session_groups.sort_values(sort_by, ascending=ascending)
    
    # Pagination
    total_sessions = len(session_groups)
    total_pages = max(1, (total_sessions + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    paginated_sessions = session_groups.iloc[start_idx:end_idx]
    
    # Format sessions for display
    sessions_list = []
    for _, row in paginated_sessions.iterrows():
        sessions_list.append({
            'session_id': row['session_id'],
            'user_id': row['user_id'],
            'event_count': int(row['event_count']),
            'first_event': row['first_event'].strftime('%Y-%m-%d %H:%M:%S'),
            'last_event': row['last_event'].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return templates.TemplateResponse(
        "sessions.html",
        {
            "request": request,
            "sessions": sessions_list,
            "total_sessions": total_sessions,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "sort_by": sort_by,
            "sort_order": sort_order
        }
    )
    
    # Get unique sessions with their stats
    session_stats = sessions_df.groupby('session_id').agg({
        'item_id': 'count',
        'timestamp': ['min', 'max']
    }).reset_index()
    
    session_stats.columns = ['session_id', 'event_count', 'first_event', 'last_event']
    session_stats = session_stats.sort_values('first_event', ascending=False).head(100)
    
    sessions = []
    for _, row in session_stats.iterrows():
        sessions.append({
            'session_id': row['session_id'],
            'event_count': int(row['event_count']),
            'first_event': row['first_event'].strftime('%Y-%m-%d %H:%M:%S'),
            'last_event': row['last_event'].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return templates.TemplateResponse(
        "sessions.html",
        {
            "request": request,
            "sessions": sessions,
            "total_sessions": sessions_df['session_id'].nunique()
        }
    )


@app.get("/session-search")
async def session_search(session_id: str):
    """Redirect to session detail page"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/session/{session_id}")


@app.post("/recommend-from-session/{session_id}", response_class=HTMLResponse)
async def recommend_from_session(
    request: Request,
    session_id: str,
    top_k: int = Form(10)
):
    """Generate recommendations based on items viewed in a session"""
    try:
        if sessions_df is None or analyzer is None:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "System not initialized"}
            )
        
        # Get session events
        session_events = sessions_df[sessions_df['session_id'] == session_id].copy()
        
        if session_events.empty:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": f"Session {session_id} not found"}
            )
        
        # Get item IDs from session
        session_items = session_events['item_id'].tolist()
        
        logger.info(f"Generating recommendations for session {session_id} with {len(session_items)} items")
        
        # Get recommendations
        recommendations = analyzer.get_recommendations(session_items, top_k=top_k)
        
        # Generate map HTML (session items + recommendations)
        map_html = analyzer.generate_map_html(session_items, recommendations)
        
        # Get session details
        session_details = []
        for item_id in session_items:
            listing = analyzer.item_to_listing.get(item_id)
            if listing:
                session_details.append({
                    'id': item_id,
                    'city': listing.get('city', 'N/A'),
                    'neighborhood': listing.get('neighborhood', 'N/A'),
                    'price': listing.get('price', 0),
                    'bedrooms': listing.get('bedrooms', 0),
                    'bathrooms': listing.get('bathrooms', 0),
                    'suites': listing.get('suites', 0),
                    'parking': listing.get('parking_spaces', 0),
                    'usable_area': listing.get('usable_areas', 0),
                    'total_area': listing.get('total_areas', 0),
                    'type': listing.get('unit_type', 'N/A'),
                    'usage': listing.get('usage_type', 'N/A'),
                    'business': listing.get('business_type', 'N/A'),
                    'status': listing.get('status', 'N/A')
                })
        
        # Get recommendation details
        rec_details = []
        for rank, (item_id, score) in enumerate(recommendations, 1):
            listing = analyzer.item_to_listing.get(item_id)
            if listing:
                # Calculate min distance from session
                min_distance = float('inf')
                for sess_id in session_items:
                    dist = analyzer.calculate_distance(sess_id, item_id)
                    if dist is not None:
                        min_distance = min(min_distance, dist)
                
                rec_details.append({
                    'rank': rank,
                    'id': item_id,
                    'score': f"{score:.4f}",
                    'city': listing.get('city', 'N/A'),
                    'neighborhood': listing.get('neighborhood', 'N/A'),
                    'price': listing.get('price', 0),
                    'bedrooms': listing.get('bedrooms', 0),
                    'bathrooms': listing.get('bathrooms', 0),
                    'suites': listing.get('suites', 0),
                    'parking': listing.get('parking_spaces', 0),
                    'usable_area': listing.get('usable_areas', 0),
                    'total_area': listing.get('total_areas', 0),
                    'type': listing.get('unit_type', 'N/A'),
                    'usage': listing.get('usage_type', 'N/A'),
                    'business': listing.get('business_type', 'N/A'),
                    'distance_km': f"{min_distance:.2f}" if min_distance != float('inf') else "N/A",
                    'status': listing.get('status', 'N/A')
                })
        
        return templates.TemplateResponse(
            "session_recommendations.html",
            {
                "request": request,
                "session_id": session_id,
                "session_items": session_items,
                "session_count": len(session_items),
                "top_k": top_k,
                "map_html": map_html,
                "session_details": session_details,
                "recommendations": rec_details
            }
        )
        
    except ValueError as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": str(e)}
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": f"Internal error: {str(e)}"}
        )


@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    """View listings in a specific session"""
    if sessions_df is None or analyzer is None:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": "Sessions data not loaded"}
        )
    
    # Get session events
    session_events = sessions_df[sessions_df['session_id'] == session_id].copy()
    
    if session_events.empty:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": f"Session {session_id} not found"}
        )
    
    # Sort by timestamp
    session_events = session_events.sort_values('timestamp')
    
    # Get item IDs from session
    session_item_ids = session_events['item_id'].tolist()
    
    # Generate map with session items only (no recommendations)
    import folium
    
    # Calculate center
    lats, lons = [], []
    for item_id in session_item_ids:
        listing = analyzer.item_to_listing.get(item_id)
        if listing and 'lat' in listing and 'lon' in listing:
            lats.append(listing['lat'])
            lons.append(listing['lon'])
    
    if lats and lons:
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    else:
        center_lat, center_lon = -14.235, -51.925
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Add session items to map
    for idx, item_id in enumerate(session_item_ids, 1):
        listing = analyzer.item_to_listing.get(item_id)
        if listing and 'lat' in listing and 'lon' in listing:
            # Circle marker azul com número da posição
            folium.CircleMarker(
                location=[listing['lat'], listing['lon']],
                radius=12,
                popup=folium.Popup(
                    f"""
                    <b>Position {idx}</b><br>
                    ID: {item_id}<br>
                    City: {listing.get('city', 'N/A')}<br>
                    Price: R$ {listing.get('price', 0):,.0f}<br>
                    Beds: {listing.get('bedrooms', 0):.0f}<br>
                    """,
                    max_width=250
                ),
                tooltip=f"Position {idx}: {listing.get('city', 'N/A')}",
                color='#2E86C1',
                fill=True,
                fillColor='#3498DB',
                fillOpacity=0.8,
                weight=2
            ).add_to(m)
            
            # Adicionar número da posição
            folium.Marker(
                location=[listing['lat'], listing['lon']],
                icon=folium.DivIcon(html=f"""
                    <div style="
                        font-size: 11px; 
                        color: white; 
                        font-weight: bold;
                        text-align: center;
                        margin-left: -6px;
                        margin-top: -6px;
                    ">{idx}</div>
                """)
            ).add_to(m)
    
    map_html = m._repr_html_()
    
    # Get listing details for each item
    viewed_listings = []
    for idx, event in session_events.iterrows():
        item_id = event['item_id']
        listing = analyzer.item_to_listing.get(item_id)
        
        if listing:
            viewed_listings.append({
                'position': int(event.get('position', 0)),
                'item_id': item_id,
                'timestamp': event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'event_type': event.get('event_type', 'N/A'),
                'city': listing.get('city', 'N/A'),
                'neighborhood': listing.get('neighborhood', 'N/A'),
                'price': listing.get('price', 0),
                'bedrooms': listing.get('bedrooms', 0),
                'bathrooms': listing.get('bathrooms', 0),
                'area': listing.get('usable_areas', 0),
                'business_type': listing.get('business_type', 'N/A')
            })
    
    return templates.TemplateResponse(
        "session_detail.html",
        {
            "request": request,
            "session_id": session_id,
            "user_id": session_events.iloc[0]['user_id'],
            "event_count": len(session_events),
            "first_event": session_events.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            "last_event": session_events.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            "viewed_listings": viewed_listings,
            "map_html": map_html
        }
    )


def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Fermi Recommendation API")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path to .pth file (e.g., GRU4Rec or path/to/model.pth)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", reload=args.reload)
