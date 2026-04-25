from datasets import load_dataset
import json

ds = load_dataset('VLSP2025-LegalSML/legal-pretrain', split='train', streaming=True)

keywords = ['giao thông', 'đường bộ', 'lái xe', 'an toàn giao thông',
            'giấy phép lái', 'phương tiện giao thông']

found = []
count = 0
for row in ds:
    count += 1
    name = row['metadata'].get('DocName', '').lower()
    if any(kw in name for kw in keywords):
        found.append({
            'identity': row['metadata'].get('DocIdentity', ''),
            'name': row['metadata'].get('DocName', ''),
            'organ': row['metadata'].get('OrganName', ''),
            'date': str(row['metadata'].get('IssueDate', ''))[:10],
            'size_kb': len(row['doc_content']) // 1024,
            'content': row['doc_content'],
        })
    if count % 20000 == 0:
        print(f'  Duyet {count}... tim thay {len(found)}', flush=True)

print(f'\nTong: {count} van ban, tim thay {len(found)}', flush=True)

with open('./data/traffic_law_docs.json', 'w', encoding='utf-8') as f:
    json.dump(found, f, ensure_ascii=False)
print(f'Da luu {len(found)} van ban vao ./data/traffic_law_docs.json', flush=True)

for i, doc in enumerate(sorted(found, key=lambda x: x['date'], reverse=True)):
    print(f'[{i+1}] {doc["identity"]} | {doc["name"][:80]} | {doc["date"]} | {doc["size_kb"]}KB', flush=True)
