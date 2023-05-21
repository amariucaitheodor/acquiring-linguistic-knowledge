# WiT

These statistics for the [WIT dataset](https://huggingface.co/datasets/theodor1289/wit) are computed [here](https://github.com/amariucaitheodor/acquiring-linguistic-knowledge/blob/ea8ef213dda6d2f2d65e1a1252564fbd9392eccd/alkmi/data/text_datamodules.py) 
using the [evaluate](https://huggingface.co/spaces/evaluate-measurement) library.

```text
(This is what we're using) Captions + alternative text associated with the images:
Split: train[:1%] ----> Total words: 1.697.524, No. of duplicates: 1.605.082, No. of unique: 92.442
Split: train[:10%] ----> Total words: 16.927.997, No. of duplicates: 16.536.584, No. of unique: 391.413
Split: train[:100%] ----> Total words: 168.994.535, No. of duplicates: 167.526.685, No. of unique: 1.467.850

(FYI) Just the captions associated with the images:
Split: train[:1%] ----> Total words: 64.710, No. of duplicates: 47.583, No. of unique: 17.127
Split: train[:10%] ----> Total words: 646.452, No. of duplicates: 565.569, No. of unique: 80.883
Split: train[:100%] ----> Total words: 6.479.627, No. of duplicates: 6.158.732, No. of unique: 320.895
```
 