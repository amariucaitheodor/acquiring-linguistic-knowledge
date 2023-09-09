from transformers import BertTokenizerFast

from alkmi.models.lightning_flava import FlavaPreTrainingLightningModule
from alkmi.models.flava import FlavaProcessor
from alkmi.models.flava import FlavaForPreTraining

MODEL_TYPE = 'half_sized'

closest_avail_model = {
    (1, 0): "half_bs4032_seed5501650_bf16-mixed/flava-epoch=00-step=4821.ckpt",
    # 4956 closest global_step to Step 4308 for text1_vision0
    (1, 1): "half_bs4096_seed5501650_bf16-mixed/flava-epoch=00-step=6397.ckpt",
    # 6349 closest global_step to Step 11242 for text1_vision1
    (10, 0): "half_bs4080_seed5501650_bf16-mixed/flava-epoch=00-step=14495.ckpt",
    # 14749 closest global_step to Step 25683 for text10_vision0
    (10, 10): "half_bs4096_seed5501650_bf16-mixed/flava-epoch=00-step=14324.ckpt",
    # 14340 closest global_step to Step 16912 for text10_vision10
}

tokenizer: BertTokenizerFast = FlavaProcessor.from_pretrained("facebook/flava-full").tokenizer
for text_perc in [1, 10]:
    for vision_perc in [0, text_perc]:
        NAME = f"bestckpt_{MODEL_TYPE}_text{text_perc}_vision{vision_perc}"
        tokenizer.save_pretrained(f"theodor1289/{NAME}", push_to_hub=True)

        path = f"/cluster/work/cotterell/tamariucai/HuggingfaceCheckpoints/flava-wit/text{text_perc}-vision{vision_perc}/{closest_avail_model[(text_perc, vision_perc)]}"
        model: FlavaForPreTraining = FlavaPreTrainingLightningModule.load_from_checkpoint(
            path,
            # arguments below are irrelevant but needed to load
            pretrained=False, half_size=True,
            learning_rate=0.001, adam_eps=1e-8, adam_weight_decay=0.01,
            adam_betas=(0.9, 0.999), warmup_steps=10000, max_steps=450000,
        ).model
        model.save_pretrained(f"theodor1289/{NAME}", push_to_hub=True, commit_message=path)
