import random

import torch


class ITMTransform:
    def __call__(self, info, dataset, itm_probability):
        output = {"itm_labels": torch.full(size=[len(info["text"])], fill_value=1.0 - itm_probability)}
        output["itm_labels"] = torch.bernoulli(input=output["itm_labels"]).long()
        for i in range(len(output["itm_labels"])):
            if output["itm_labels"][i] == 0:
                original = info["text"][i]
                while info["text"][i] == original:  # rejection sampling
                    info["text"][i] = dataset.select([random.randint(0, len(dataset) - 1)])[0]["text"]
        output.update({"image": info["image"]})  # Original image order is kept, text could change (ITM)
        output.update({"text": info["text"]})
        return output
