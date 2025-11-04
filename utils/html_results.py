from pathlib import Path
import base64
from datetime import datetime

RESULTS_DIR = Path("results")
TEMPLATE_DIR = Path("templates")
TEMPLATE_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = TEMPLATE_DIR / "results.html"

def generate_html():
    # RECURSIVE SEARCH: **all files in subfolders**
    files = sorted(
        RESULTS_DIR.rglob("*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    # Filter: Filter only files (skip folders)
    files = [f for f in files if f.is_file()]

    html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breast Cancer Model Results</title>
    <style>
        body {{font-family: 'Segoe UI', sans-serif; background: #f4f7fa; color: #333; margin: 0; padding: 20px;}}
        .container {{max-width: 1100px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);}}
        h1 {{text-align: center; color: #2c3e50;}}
        .path {{font-family: monospace; color: #7f8c8d; font-size: 0.9em;}}
        table {{width: 100%; border-collapse: collapse; margin-top: 20px;}}
        th, td {{padding: 12px; text-align: left; border-bottom: 1px solid #ddd;}}
        th {{background: #3498db; color: white;}}
        tr:hover {{background: #f8f9fa;}}
        .download-btn {{background: #27ae60; color: white; padding: 6px 12px; text-decoration: none; border-radius: 4px; font-size: 0.9em;}}
        .download-btn:hover {{background: #219653;}}
        .image-preview {{max-width: 180px; max-height: 130px; border-radius: 6px;}}
        .footer {{text-align: center; margin-top: 40px; color: #95a5a6; font-size: 0.9em;}}
    </style>
</head>
<body>
    <div class="container">
        <h1>Breast Cancer Model Results</h1>
        <p><strong>Total of files:</strong> {len(files)} (including sub folders)</p>
        
        <table>
            <thead>
                <tr>
                    <th>Path</th>
                    <th>File</th>
                    <th>Type</th>
                    <th>Size</th>
                    <th>Modification</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
    """

    for file in files:
        rel_path = file.relative_to(RESULTS_DIR) 
        parent_dir = str(rel_path.parent) if rel_path.parent != Path(".") else "raiz"
        size = file.stat().st_size
        size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
        mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
        
        ext = file.suffix.lower()
        file_type = "Unknown"
        preview = ""

        if ext in ['.png', '.jpg', '.jpeg', '.gif']:
            file_type = "Imagem"
            try:
                with open(file, "rb") as img:
                    encoded = base64.b64encode(img.read()).decode()
                preview = f'<img src="data:image/png;base64,{encoded}" class="image-preview" alt="{file.name}">'
            except:
                preview = "<em>Error to load</em>"
        elif ext == '.csv':
            file_type = "CSV"
        elif ext == '.pkl':
            file_type = "Modelo"
        elif ext == '.json':
            file_type = "JSON"
        elif ext == '.txt':
            file_type = "Texto"

        download_url = f"/results/{rel_path.as_posix()}"
        
        html_content += f"""
            <tr>
                <td><span class="path">{parent_dir}</span></td>
                <td><strong>{file.name}</strong></td>
                <td>{file_type}</td>
                <td>{size_str}</td>
                <td>{mod_time}</td>
                <td>
                    <a href="{download_url}" class="download-btn" download>Download</a>
                    {f'<br>{preview}' if preview else ''}
                </td>
            </tr>
        """

    html_content += f"""
            </tbody>
        </table>
        <div class="footer">
            <p>Generated on {datetime.now().strftime("%d/%m/%Y %H:%M")} | Contains sub folders</p>
        </div>
    </div>
</body>
</html>
    """

    OUTPUT_FILE.write_text(html_content, encoding='utf-8')
    print(f"HTML generated: {OUTPUT_FILE} ({len(files)} files)")

if __name__ == "__main__":
    generate_html()