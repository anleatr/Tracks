from pathlib import Path

mock_data_file_path = Path("./examples/mock_data.txt")
    
with open(mock_data_file_path, "r", encoding="utf-8") as file:
    texts = file.read()

