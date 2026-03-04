import os
import sys
from datetime import datetime

def record_session(cli_name, title, content, base_dir=None):
    if base_dir is None:
        # Default to ~/daily_work to be user-agnostic
        base_dir = os.path.join(os.path.expanduser("~"), "daily_work")
    
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    target_dir = os.path.join(base_dir, date_str)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    file_name = f"{cli_name}-session.md"
    file_path = os.path.join(target_dir, file_name)
    
    formatted_content = f"""
---
## {cli_name.capitalize()} Session - {now.strftime("%H:%M:%S")}

### {title}

{content}
"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(formatted_content)
    print(f"Successfully recorded session to {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(1)
    # Support optional 4th argument for base_dir
    b_dir = sys.argv[4] if len(sys.argv) > 4 else None
    record_session(sys.argv[1], sys.argv[2], sys.argv[3], b_dir)
