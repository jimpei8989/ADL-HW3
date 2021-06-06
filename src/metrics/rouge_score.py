def flatten_dict(d, prefixes=()):
    ret = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ret |= flatten_dict(v, prefixes=prefixes + (k,))
        elif isinstance(v, (int, float)):
            ret |= {"_".join(prefixes + (k,)): v}
        else:
            raise ValueError
    return ret


class RougeScore:
    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer

    def decode(self, tokens):
        text = self.tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
        stop_index = text.find(self.tokenizer.eos_token)

        if stop_index is not None:
            text = text[:stop_index]

        return text

    def __call__(self, eval_predictions):
        pred_texts = [self.decode(p) for p in eval_predictions.predictions]
        label_texts = [self.decode(l) for l in eval_predictions.label_ids]

        from tw_rouge import get_rouge

        rouge = get_rouge(pred_texts, label_texts, avg=True, ignore_empty=False)
        return flatten_dict(rouge)
