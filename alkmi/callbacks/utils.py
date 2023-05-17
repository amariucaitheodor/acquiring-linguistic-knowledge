from transformers import PreTrainedModel, RobertaTokenizerFast, BertTokenizerFast, RobertaForMaskedLM, \
    PreTrainedTokenizerBase


def get_corresponding_tokenizer_for_model(model: PreTrainedModel) -> PreTrainedTokenizerBase:
    if type(model) == RobertaForMaskedLM:
        return RobertaTokenizerFast.from_pretrained('roberta-base')
    return BertTokenizerFast.from_pretrained('bert-base-uncased')
