from datetime import datetime


def log(message: str, divisor_line: bool = False, jump_line: bool = False) -> None:
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if divisor_line:
        print("\n" + "=" * 50)
    if jump_line:
        print(f"\n[{timestamp}] {message}")
    else:
        print(f"[{timestamp}] {message}")

