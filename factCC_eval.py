from transformers import BertForSequenceClassification, BertTokenizer

model_path = 'manueldeprada/FactCC'

tokenizer = BertTokenizer.from_pretrained(model_path)

model = BertForSequenceClassification.from_pretrained(model_path)

text=['fortune cookies originated in san francisco', 'fortune cookies originated in california', 'fortune cookies originated in us', 'fortune cookies were made by san francisco bakery', 'fortune cookies were made by californian bakery', 'precise origin of fortune cookies is unclear']
response = '''fortune cookies originated in a specific region of japan'''

for text in text:
    input_dict = tokenizer(text, response, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
    logits = model(**input_dict).logits
    pred = logits.argmax(dim=1)
    print(model.config.id2label[pred.item()])