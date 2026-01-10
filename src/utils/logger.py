from datetime import datetime


def log(message: str, jump_line: bool = False) -> None:
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if jump_line:
        print(f"\n[{timestamp}] {message}")
    else:
        print(f"[{timestamp}] {message}")

