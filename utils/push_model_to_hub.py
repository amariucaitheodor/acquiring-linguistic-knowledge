from transformers import BertTokenizerFast

from alkmi.models.lightning_flava import FlavaPreTrainingLightningModule
from alkmi.models.flava import FlavaProcessor
from alkmi.models.flava import FlavaForPreTraining

NAME = "flava_text_only_2"
CKPT_LOCATION = "text10-vision0/bs4096_seed5501650_bf16-mixed/last.ckpt"

tokenizer: BertTokenizerFast = FlavaProcessor.from_pretrained("facebook/flava-full").tokenizer
tokenizer.save_pretrained(f"theodor1289/{NAME}", push_to_hub=True)

model: FlavaForPreTraining = FlavaPreTrainingLightningModule.load_from_checkpoint(
    f"/cluster/work/cotterell/tamariucai/HuggingfaceCheckpoints/flava-wit/{CKPT_LOCATION}",
    # arguments below are irrelevant
    pretrained=False, half_size=False,
    learning_rate=0.001, adam_eps=1e-8, adam_weight_decay=0.01,
    adam_betas=(0.9, 0.999), warmup_steps=10000, max_steps=450000,
).model
model.save_pretrained(f"theodor1289/{NAME}", push_to_hub=True)
