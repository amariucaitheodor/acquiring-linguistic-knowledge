from torch._dynamo import OptimizedModule
from transformers import PreTrainedModel, RobertaTokenizerFast, BertTokenizerFast, RobertaForMaskedLM, \
    PreTrainedTokenizerBase

from alkmi.models.flava import FlavaForPreTraining


def get_corresponding_tokenizer_for_model(model: PreTrainedModel) -> PreTrainedTokenizerBase:
    if type(model) == RobertaForMaskedLM:
        return RobertaTokenizerFast.from_pretrained('roberta-base')
    return BertTokenizerFast.from_pretrained('bert-base-uncased')


def replace_flava_submodel_with_orig_for_eval(model: FlavaForPreTraining) -> OptimizedModule:
    if type(model.flava.text_model) != OptimizedModule:
        print("FLAVA text model is not an optimized module! Are you using an externally pretrained model?")
        return model.flava.text_model
    optimized_text_model: OptimizedModule = model.flava.text_model
    model.flava.text_model = model.flava.text_model._orig_mod
    return optimized_text_model
