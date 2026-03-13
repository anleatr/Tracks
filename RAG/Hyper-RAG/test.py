import json
import pprint

with open("caches/onetry/kv_store_text_chunks.json", 'r') as file:
    data = json.load(file)

# sample = list(data.keys())[0]
# detail = data[sample]['content']
# print(type(detail))
# # pprint.pprint(detail)
# print(detail)

# print("Keys:", list(data.keys())[:5])

for k, v in data.items():
  print(f"Keys: {k}, Value:{v}")
  break