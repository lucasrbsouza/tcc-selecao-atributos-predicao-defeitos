"""Baixa os dados usados no experimento e confere o SHA-256 de cada arquivo (data_manifest.json)."""
import hashlib, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for item in json.loads((ROOT / 'data_manifest.json').read_text()):
    dst = ROOT / 'data' / item['file']
    if not dst.exists():
        url = item['source'].replace('github.com/hosseinkshvrz/apachejit/blob/', 'raw.githubusercontent.com/hosseinkshvrz/apachejit/')
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    assert digest == item['sha256'], f'hash diferente em {dst}'
    print('ok', item['file'])
