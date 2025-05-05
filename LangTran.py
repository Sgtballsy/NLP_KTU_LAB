from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, source_lang='en', target_lang='fr'):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    text_to_translate = "Hello, how are you?"
    translated_text = translate_text(text_to_translate, source_lang='en', target_lang='fr')
    print("Translated Text:", translated_text)