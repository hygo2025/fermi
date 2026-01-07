import os

def listings_raw_path():
    return os.getenv("LISTINGS_RAW_PATH")

def events_raw_path():
    return os.getenv("EVENTS_RAW_PATH")

def events_processed_path():
    return os.getenv("EVENTS_PROCESSED_PATH")

def listings_processed_path():
    return os.getenv("LISTINGS_PROCESSED_PATH")

def user_sessions_path():
    return os.getenv("USER_SESSIONS_PATH")

def listing_id_mapping_path():
    return os.getenv("LISTING_ID_MAPPING_PATH")

def user_id_mapping_path():
    return os.getenv("USER_ID_MAPPING_PATH")


def session_id_mapping_path():
    return os.getenv("SESSION_ID_MAPPING_PATH")
