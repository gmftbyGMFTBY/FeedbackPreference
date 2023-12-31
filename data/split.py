import json
import random


with open('processed_feedback_preference.json') as f:
    data = json.load(f)

random.seed(0)
index = random.sample(range(len(data)), 500)
test_set = [data[i] for i in index]
index = set(index)
train_set = [data[i] for i in range(len(data)) if i not in index]
print(f'[!] collect {len(test_set)} test samples and {len(train_set)} train samples')

with open('train.json', 'w') as f:
    json.dump(train_set, f, ensure_ascii=False, indent=4)

with open('test.json', 'w') as f:
    json.dump(test_set, f, ensure_ascii=False, indent=4)
