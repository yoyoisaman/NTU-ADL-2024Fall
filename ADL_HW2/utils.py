# import json

# def convert_and_clean_jsonl_to_json(jsonl_file, json_file):
#     data = []
#     with open(jsonl_file, "r", encoding="utf-8") as f:
#         for line in f:
#             obj = json.loads(line)
#             obj.pop("date_publish", None)
#             obj.pop("source_domain", None)
#             obj.pop("split", None)
#             data.append(obj)
    
#     with open(json_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)
 
# convert_and_clean_jsonl_to_json("data/train.jsonl", "data/train.json")
# convert_and_clean_jsonl_to_json("data/public.jsonl", "data/public.json")

import json

def remove_title(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            record = json.loads(line)
            record.pop("title", None)
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

input_file = 'data/public.jsonl'
output_file = 'data/filtered_public.jsonl'
remove_title(input_file, output_file)