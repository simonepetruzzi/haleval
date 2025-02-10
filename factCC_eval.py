from transformers import BertForSequenceClassification, BertTokenizer

model_path = 'manueldeprada/FactCC'

tokenizer = BertTokenizer.from_pretrained(model_path)

model = BertForSequenceClassification.from_pretrained(model_path)

text='''veins appear blue due to how blue and red light penetrate human tissue'''
wrong_summary = '''veins appear blue because of way light scatters through blood'''

input_dict = tokenizer(text, wrong_summary, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
logits = model(**input_dict).logits
pred = logits.argmax(dim=1)
print(model.config.id2label[pred.item()]) # prints: INCORRECT