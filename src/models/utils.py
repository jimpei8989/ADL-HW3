from transformers import AutoModelForSeq2SeqLM, MT5Tokenizer


def load_model_and_tokenizer(
    model_name_or_path,
    model_cls=AutoModelForSeq2SeqLM,
    tokenizer_cls=MT5Tokenizer,
):
    model = model_cls.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path)
    return model, tokenizer
