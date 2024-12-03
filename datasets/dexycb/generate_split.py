import json

with open('/home/zhouyaqing/gSDF/datasets/dexycb/dexycb_test_s2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
ids = [str(item['id']).zfill(6) for item in data['images']]

with open('test_s2.json', 'w', encoding='utf-8') as f:
    json.dump(ids, f, ensure_ascii=False, indent=4)