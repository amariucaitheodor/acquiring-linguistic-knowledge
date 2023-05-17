import importlib
from itertools import chain

import torch
from transformers.modeling_outputs import MaskedLMOutput

from callbacks.utils import get_corresponding_tokenizer_for_model
from alkmi.definitions import TEXT_MAX_LENGTH_DEFAULT

lm_eval = importlib.import_module(name="lm_eval", package="lm-evaluation-harness")

from lm_eval.api import utils
from lm_eval.models.huggingface import AutoMaskedLM
from tqdm import tqdm
from transformers import PreTrainedModel
from typing import List, Optional, Tuple, Union


class TextLM(AutoMaskedLM):
    """Implements a language model interface for the BERT/RoBERTa/etc. text PyTorch models.
    See: https://github.com/aaronmueller/lm-evaluation-harness
    """

    def __init__(
            self,
            model: PreTrainedModel,
            batch_size: Optional[int] = 1,
            max_length: Optional[int] = TEXT_MAX_LENGTH_DEFAULT,
            add_special_tokens: Optional[bool] = None,
            device: Optional[Union[int, str]] = "cuda",
    ):
        self._batch_size = batch_size
        self._max_length = max_length

        self._add_special_tokens = add_special_tokens
        self.tokenizer = get_corresponding_tokenizer_for_model(model)
        self.tokenizer.model_max_length = self.max_length

        self.model = model
        self._device = device

    def loglikelihood(
            self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """
        Returns *pseudo*-loglikelihoods, as described in Salazar et al. (2020).
        """
        scores = []
        for chunk in utils.chunks(tqdm(requests, disable=False), self.batch_size):
            _, continuation = zip(*chunk)

            tokenized = self._prepare_text(continuation)

            token_ids, attention_masks, effective_token_ids, lengths, offsets = list(zip(*tokenized))
            token_ids = torch.cat(token_ids).to(self.device)
            attention_masks = torch.cat(attention_masks).to(self.device).bool()
            effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

            indices = list(chain.from_iterable([list(range(o, o + n)) for n, o in zip(lengths, offsets)]))

            with torch.inference_mode():
                output: MaskedLMOutput = self.model(input_ids=token_ids, attention_mask=attention_masks)
                logits = output.logits.detach()[torch.arange(sum(lengths)), indices]

            logprob_distribution = logits - logits.logsumexp(1).unsqueeze(1)
            logprob_distribution = logprob_distribution / torch.tensor(2).log()

            batch_scores = logprob_distribution[torch.arange(sum(lengths)), effective_token_ids].type(
                torch.DoubleTensor).split(lengths)
            batch_scores = [(float(s.sum()),) for s in batch_scores]

            scores.extend(batch_scores)
        return scores
