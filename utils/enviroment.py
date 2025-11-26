import os


def concat_paths(*paths):
    return os.path.join(*paths)

def get_config(base_path: str):
    events_processed_path = 'processed_data/events'
    listings_processed_path = 'processed_data/listings'
    enriched_events_path = 'processed_data/enriched_events'

    return {
        "EVENTS_PROCESSED_PATH": concat_paths(base_path, events_processed_path),
        "LISTINGS_PROCESSED_PATH": concat_paths(base_path, listings_processed_path),
        "ENRICHED_EVENTS_PATH": concat_paths(base_path, enriched_events_path),
    }