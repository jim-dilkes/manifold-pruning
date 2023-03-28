import json
from sklearn.model_selection import train_test_split

with open('train-v2.0.json', 'r') as f:
    train_val_raw = json.load(f)

with open('dev-v2.0.json', 'r') as f:
    test_raw = json.load(f)

train_val_qa = []
for data in train_val_raw['data']:
    for par in data['paragraphs']:
        for question in par['qas']:
            train_val_qa.append(
                [question['question'], question['is_impossible']])

test_qa = []
for data in test_raw['data']:
    for par in data['paragraphs']:
        for question in par['qas']:
            test_qa.append([question['question'], question['is_impossible']])


train_qa, val_qa = train_test_split(
    train_val_qa, test_size=0.05, shuffle=False, random_state=999)

for mode in ('Train', 'Test', 'Validation'):
    file_base = mode+'_'
    if mode == 'Train':
        question, adversarial = zip(*train_qa)
    if mode == 'Validation':
        question, adversarial = zip(*val_qa)
    if mode == 'Test':
        question, adversarial = zip(*test_qa)
    adversarial_copy = list(adversarial).copy()
    adversarial = [str(adv)+'\n' for adv in adversarial]
    question = [q+'\n' for q in question]

    with open(mode+'/'+file_base+'QA.txt', 'w', newline='\n') as f:
        f.writelines(question)
    with open(mode+'/'+file_base+'answerable.txt', 'w', newline='\n') as f:
        f.writelines(adversarial)

    adversarial_questions = []
    simple_questions = []
    for q, adv in zip(question, adversarial_copy):
        if adv:
            adversarial_questions.append(q)
        else:
            simple_questions.append(q)
    with open(mode+'/'+file_base+'QA_adversarial.txt', 'w') as f:
        f.writelines(adversarial_questions)
    with open(mode+'/'+file_base+'QA_simple.txt', 'w') as f:
        f.writelines(simple_questions)
