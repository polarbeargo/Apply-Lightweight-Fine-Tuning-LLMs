# Lightweight Fine-Tuning Project
<!-- embedd output.png and output.png -->
[image1]: ./images/output.png
[image2]: ./images/gpt2train.png
[image3]: ./images/gpt2RunHistory.png
[image4]: ./images/GPT2Summary.png
[image5]: ./images/gpt2train.png
[image6]: ./images/lorahistorygpt2.png
[image7]: ./images/loragpt2Summary.png

## Environment Setup

- Run the following commands to setup the local environment:

```
chmod +x setupLightweightFineTuningLLMs.sh  
./setupLightweightFineTuningLLMs.sh
```
- You can also directly execute the kaggle notebooks from the links provided below.

## Fine-Tuning

We use small datasets to fine-tune the models as prototypes.

## Results  

#### GPT2 Model Architecture

```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```

![][image2]  
History      |  Summary
:-------------------------:|:-------------------------:
![][image3]                | ![][image4]  

- [Wandb Report Link](https://wandb.ai/bow1226/Fine%20tuning%20gpt2/reports/GPT2-QLoRA---Vmlldzo3MDAyMTE0)

- [kaggle notebook link](https://www.kaggle.com/code/hsinwenchang/lightweightfinetuningnlp-ipynb)

#### After PEFT QLoRA Model Architecture

```
trainable params: 2,359,296 || all params: 126,799,104 || trainable%: 1.8606566809809635
None
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-11): 12 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2Attention(
              (c_attn): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=768, out_features=2304, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=768, out_features=64, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=64, out_features=2304, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (c_proj): Linear4bit(in_features=768, out_features=768, bias=True)
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): Linear4bit(in_features=768, out_features=3072, bias=True)
              (c_proj): Linear4bit(in_features=3072, out_features=768, bias=True)
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )
  )
)
```

![][image5]
History      |  Summary
:-------------------------:|:-------------------------:
![][image6]                | ![][image7]  

- [Wandb Report Link](https://api.wandb.ai/links/bow1226/r6g4akdw)
- [kaggle notebook link](https://www.kaggle.com/code/hsinwenchang/lightweightfinetuningnlpqlora-ipynb)

- [Saved PEFT Model directory](https://github.com/polarbeargo/Apply-Lightweight-Fine-Tuning-LLMs/tree/main/gpt2-lora):

#### nvidia/mit-b0 Model Architecture

```
SegformerForSemanticSegmentation(
  (segformer): SegformerModel(
    (encoder): SegformerEncoder(
      (patch_embeddings): ModuleList(
        (0): SegformerOverlapPatchEmbeddings(
          (proj): Conv2d(3, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
          (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        )
        (1): SegformerOverlapPatchEmbeddings(
          (proj): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        )
        (2): SegformerOverlapPatchEmbeddings(
          (proj): Conv2d(64, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (layer_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
        )
        (3): SegformerOverlapPatchEmbeddings(
          (proj): Conv2d(160, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
      )
      (block): ModuleList(
        (0): ModuleList(
          (0): SegformerLayer(
            (layer_norm_1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            (attention): SegformerAttention(
              (self): SegformerEfficientSelfAttention(
                (query): Linear(in_features=32, out_features=32, bias=True)
                (key): Linear(in_features=32, out_features=32, bias=True)
                (value): Linear(in_features=32, out_features=32, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (sr): Conv2d(32, 32, kernel_size=(8, 8), stride=(8, 8))
                (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
              )
              (output): SegformerSelfOutput(
                (dense): Linear(in_features=32, out_features=32, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): Identity()
            (layer_norm_2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            (mlp): SegformerMixFFN(
              (dense1): Linear(in_features=32, out_features=128, bias=True)
              (dwconv): SegformerDWConv(
                (dwconv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
              )
              (intermediate_act_fn): GELUActivation()
              (dense2): Linear(in_features=128, out_features=32, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SegformerLayer(
            (layer_norm_1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            (attention): SegformerAttention(
              (self): SegformerEfficientSelfAttention(
                (query): Linear(in_features=32, out_features=32, bias=True)
                (key): Linear(in_features=32, out_features=32, bias=True)
                (value): Linear(in_features=32, out_features=32, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (sr): Conv2d(32, 32, kernel_size=(8, 8), stride=(8, 8))
                (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
              )
              (output): SegformerSelfOutput(
                (dense): Linear(in_features=32, out_features=32, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): SegformerDropPath(p=0.014285714365541935)
            (layer_norm_2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            (mlp): SegformerMixFFN(
              (dense1): Linear(in_features=32, out_features=128, bias=True)
              (dwconv): SegformerDWConv(
                (dwconv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
              )
              (intermediate_act_fn): GELUActivation()
              (dense2): Linear(in_features=128, out_features=32, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (1): ModuleList(
          (0): SegformerLayer(
            (layer_norm_1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (attention): SegformerAttention(
              (self): SegformerEfficientSelfAttention(
                (query): Linear(in_features=64, out_features=64, bias=True)
                (key): Linear(in_features=64, out_features=64, bias=True)
                (value): Linear(in_features=64, out_features=64, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (sr): Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4))
                (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
              )
              (output): SegformerSelfOutput(
                (dense): Linear(in_features=64, out_features=64, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): SegformerDropPath(p=0.02857142873108387)
            (layer_norm_2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (mlp): SegformerMixFFN(
              (dense1): Linear(in_features=64, out_features=256, bias=True)
              (dwconv): SegformerDWConv(
                (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
              )
              (intermediate_act_fn): GELUActivation()
              (dense2): Linear(in_features=256, out_features=64, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SegformerLayer(
            (layer_norm_1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (attention): SegformerAttention(
              (self): SegformerEfficientSelfAttention(
                (query): Linear(in_features=64, out_features=64, bias=True)
                (key): Linear(in_features=64, out_features=64, bias=True)
                (value): Linear(in_features=64, out_features=64, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (sr): Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4))
                (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
              )
              (output): SegformerSelfOutput(
                (dense): Linear(in_features=64, out_features=64, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): SegformerDropPath(p=0.04285714402794838)
            (layer_norm_2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (mlp): SegformerMixFFN(
              (dense1): Linear(in_features=64, out_features=256, bias=True)
              (dwconv): SegformerDWConv(
                (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
              )
              (intermediate_act_fn): GELUActivation()
              (dense2): Linear(in_features=256, out_features=64, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (2): ModuleList(
          (0): SegformerLayer(
            (layer_norm_1): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
            (attention): SegformerAttention(
              (self): SegformerEfficientSelfAttention(
                (query): Linear(in_features=160, out_features=160, bias=True)
                (key): Linear(in_features=160, out_features=160, bias=True)
                (value): Linear(in_features=160, out_features=160, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (sr): Conv2d(160, 160, kernel_size=(2, 2), stride=(2, 2))
                (layer_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
              )
              (output): SegformerSelfOutput(
                (dense): Linear(in_features=160, out_features=160, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): SegformerDropPath(p=0.05714285746216774)
            (layer_norm_2): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
            (mlp): SegformerMixFFN(
              (dense1): Linear(in_features=160, out_features=640, bias=True)
              (dwconv): SegformerDWConv(
                (dwconv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640)
              )
              (intermediate_act_fn): GELUActivation()
              (dense2): Linear(in_features=640, out_features=160, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SegformerLayer(
            (layer_norm_1): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
            (attention): SegformerAttention(
              (self): SegformerEfficientSelfAttention(
                (query): Linear(in_features=160, out_features=160, bias=True)
                (key): Linear(in_features=160, out_features=160, bias=True)
                (value): Linear(in_features=160, out_features=160, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (sr): Conv2d(160, 160, kernel_size=(2, 2), stride=(2, 2))
                (layer_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
              )
              (output): SegformerSelfOutput(
                (dense): Linear(in_features=160, out_features=160, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): SegformerDropPath(p=0.0714285746216774)
            (layer_norm_2): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
            (mlp): SegformerMixFFN(
              (dense1): Linear(in_features=160, out_features=640, bias=True)
              (dwconv): SegformerDWConv(
                (dwconv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640)
              )
              (intermediate_act_fn): GELUActivation()
              (dense2): Linear(in_features=640, out_features=160, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (3): ModuleList(
          (0): SegformerLayer(
            (layer_norm_1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (attention): SegformerAttention(
              (self): SegformerEfficientSelfAttention(
                (query): Linear(in_features=256, out_features=256, bias=True)
                (key): Linear(in_features=256, out_features=256, bias=True)
                (value): Linear(in_features=256, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (output): SegformerSelfOutput(
                (dense): Linear(in_features=256, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): SegformerDropPath(p=0.08571428805589676)
            (layer_norm_2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (mlp): SegformerMixFFN(
              (dense1): Linear(in_features=256, out_features=1024, bias=True)
              (dwconv): SegformerDWConv(
                (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
              )
              (intermediate_act_fn): GELUActivation()
              (dense2): Linear(in_features=1024, out_features=256, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SegformerLayer(
            (layer_norm_1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (attention): SegformerAttention(
              (self): SegformerEfficientSelfAttention(
                (query): Linear(in_features=256, out_features=256, bias=True)
                (key): Linear(in_features=256, out_features=256, bias=True)
                (value): Linear(in_features=256, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (output): SegformerSelfOutput(
                (dense): Linear(in_features=256, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (drop_path): SegformerDropPath(p=0.10000000149011612)
            (layer_norm_2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (mlp): SegformerMixFFN(
              (dense1): Linear(in_features=256, out_features=1024, bias=True)
              (dwconv): SegformerDWConv(
                (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
              )
              (intermediate_act_fn): GELUActivation()
              (dense2): Linear(in_features=1024, out_features=256, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
      (layer_norm): ModuleList(
        (0): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (2): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
        (3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (decode_head): SegformerDecodeHead(
    (linear_c): ModuleList(
      (0): SegformerMLP(
        (proj): Linear(in_features=32, out_features=256, bias=True)
      )
      (1): SegformerMLP(
        (proj): Linear(in_features=64, out_features=256, bias=True)
      )
      (2): SegformerMLP(
        (proj): Linear(in_features=160, out_features=256, bias=True)
      )
      (3): SegformerMLP(
        (proj): Linear(in_features=256, out_features=256, bias=True)
      )
    )
    (linear_fuse): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batch_norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activation): ReLU()
    (dropout): Dropout(p=0.1, inplace=False)
    (classifier): Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

#### After PEFT QLoRA nvidia/mit-b0 Model Architecture

```
PeftModel(
  (base_model): LoraModel(
    (model): SegformerForSemanticSegmentation(
      (segformer): SegformerModel(
        (encoder): SegformerEncoder(
          (patch_embeddings): ModuleList(
            (0): SegformerOverlapPatchEmbeddings(
              (proj): Conv2d(3, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
              (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            )
            (1): SegformerOverlapPatchEmbeddings(
              (proj): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (2): SegformerOverlapPatchEmbeddings(
              (proj): Conv2d(64, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (layer_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
            )
            (3): SegformerOverlapPatchEmbeddings(
              (proj): Conv2d(160, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (block): ModuleList(
            (0): ModuleList(
              (0): SegformerLayer(
                (layer_norm_1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (attention): SegformerAttention(
                  (self): SegformerEfficientSelfAttention(
                    (query): lora.Linear(
                      (base_layer): Linear(in_features=32, out_features=32, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=32, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=32, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (key): Linear(in_features=32, out_features=32, bias=True)
                    (value): lora.Linear(
                      (base_layer): Linear(in_features=32, out_features=32, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=32, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=32, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (dropout): Dropout(p=0.0, inplace=False)
                    (sr): Conv2d(32, 32, kernel_size=(8, 8), stride=(8, 8))
                    (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                  )
                  (output): SegformerSelfOutput(
                    (dense): Linear(in_features=32, out_features=32, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                )
                (drop_path): Identity()
                (layer_norm_2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (mlp): SegformerMixFFN(
                  (dense1): Linear(in_features=32, out_features=128, bias=True)
                  (dwconv): SegformerDWConv(
                    (dwconv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
                  )
                  (intermediate_act_fn): GELUActivation()
                  (dense2): Linear(in_features=128, out_features=32, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (1): SegformerLayer(
                (layer_norm_1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (attention): SegformerAttention(
                  (self): SegformerEfficientSelfAttention(
                    (query): lora.Linear(
                      (base_layer): Linear(in_features=32, out_features=32, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=32, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=32, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (key): Linear(in_features=32, out_features=32, bias=True)
                    (value): lora.Linear(
                      (base_layer): Linear(in_features=32, out_features=32, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=32, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=32, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (dropout): Dropout(p=0.0, inplace=False)
                    (sr): Conv2d(32, 32, kernel_size=(8, 8), stride=(8, 8))
                    (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                  )
                  (output): SegformerSelfOutput(
                    (dense): Linear(in_features=32, out_features=32, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                )
                (drop_path): SegformerDropPath(p=0.014285714365541935)
                (layer_norm_2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (mlp): SegformerMixFFN(
                  (dense1): Linear(in_features=32, out_features=128, bias=True)
                  (dwconv): SegformerDWConv(
                    (dwconv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
                  )
                  (intermediate_act_fn): GELUActivation()
                  (dense2): Linear(in_features=128, out_features=32, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (1): ModuleList(
              (0): SegformerLayer(
                (layer_norm_1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                (attention): SegformerAttention(
                  (self): SegformerEfficientSelfAttention(
                    (query): lora.Linear(
                      (base_layer): Linear(in_features=64, out_features=64, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=64, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=64, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (key): Linear(in_features=64, out_features=64, bias=True)
                    (value): lora.Linear(
                      (base_layer): Linear(in_features=64, out_features=64, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=64, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=64, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (dropout): Dropout(p=0.0, inplace=False)
                    (sr): Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4))
                    (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                  )
                  (output): SegformerSelfOutput(
                    (dense): Linear(in_features=64, out_features=64, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                )
                (drop_path): SegformerDropPath(p=0.02857142873108387)
                (layer_norm_2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                (mlp): SegformerMixFFN(
                  (dense1): Linear(in_features=64, out_features=256, bias=True)
                  (dwconv): SegformerDWConv(
                    (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
                  )
                  (intermediate_act_fn): GELUActivation()
                  (dense2): Linear(in_features=256, out_features=64, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (1): SegformerLayer(
                (layer_norm_1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                (attention): SegformerAttention(
                  (self): SegformerEfficientSelfAttention(
                    (query): lora.Linear(
                      (base_layer): Linear(in_features=64, out_features=64, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=64, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=64, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (key): Linear(in_features=64, out_features=64, bias=True)
                    (value): lora.Linear(
                      (base_layer): Linear(in_features=64, out_features=64, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=64, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=64, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (dropout): Dropout(p=0.0, inplace=False)
                    (sr): Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4))
                    (layer_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                  )
                  (output): SegformerSelfOutput(
                    (dense): Linear(in_features=64, out_features=64, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                )
                (drop_path): SegformerDropPath(p=0.04285714402794838)
                (layer_norm_2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                (mlp): SegformerMixFFN(
                  (dense1): Linear(in_features=64, out_features=256, bias=True)
                  (dwconv): SegformerDWConv(
                    (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
                  )
                  (intermediate_act_fn): GELUActivation()
                  (dense2): Linear(in_features=256, out_features=64, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (2): ModuleList(
              (0): SegformerLayer(
                (layer_norm_1): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
                (attention): SegformerAttention(
                  (self): SegformerEfficientSelfAttention(
                    (query): lora.Linear(
                      (base_layer): Linear(in_features=160, out_features=160, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=160, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=160, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (key): Linear(in_features=160, out_features=160, bias=True)
                    (value): lora.Linear(
                      (base_layer): Linear(in_features=160, out_features=160, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=160, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=160, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (dropout): Dropout(p=0.0, inplace=False)
                    (sr): Conv2d(160, 160, kernel_size=(2, 2), stride=(2, 2))
                    (layer_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
                  )
                  (output): SegformerSelfOutput(
                    (dense): Linear(in_features=160, out_features=160, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                )
                (drop_path): SegformerDropPath(p=0.05714285746216774)
                (layer_norm_2): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
                (mlp): SegformerMixFFN(
                  (dense1): Linear(in_features=160, out_features=640, bias=True)
                  (dwconv): SegformerDWConv(
                    (dwconv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640)
                  )
                  (intermediate_act_fn): GELUActivation()
                  (dense2): Linear(in_features=640, out_features=160, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (1): SegformerLayer(
                (layer_norm_1): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
                (attention): SegformerAttention(
                  (self): SegformerEfficientSelfAttention(
                    (query): lora.Linear(
                      (base_layer): Linear(in_features=160, out_features=160, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=160, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=160, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (key): Linear(in_features=160, out_features=160, bias=True)
                    (value): lora.Linear(
                      (base_layer): Linear(in_features=160, out_features=160, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=160, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=160, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (dropout): Dropout(p=0.0, inplace=False)
                    (sr): Conv2d(160, 160, kernel_size=(2, 2), stride=(2, 2))
                    (layer_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
                  )
                  (output): SegformerSelfOutput(
                    (dense): Linear(in_features=160, out_features=160, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                )
                (drop_path): SegformerDropPath(p=0.0714285746216774)
                (layer_norm_2): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
                (mlp): SegformerMixFFN(
                  (dense1): Linear(in_features=160, out_features=640, bias=True)
                  (dwconv): SegformerDWConv(
                    (dwconv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640)
                  )
                  (intermediate_act_fn): GELUActivation()
                  (dense2): Linear(in_features=640, out_features=160, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (3): ModuleList(
              (0): SegformerLayer(
                (layer_norm_1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (attention): SegformerAttention(
                  (self): SegformerEfficientSelfAttention(
                    (query): lora.Linear(
                      (base_layer): Linear(in_features=256, out_features=256, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=256, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=256, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (key): Linear(in_features=256, out_features=256, bias=True)
                    (value): lora.Linear(
                      (base_layer): Linear(in_features=256, out_features=256, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=256, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=256, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                  (output): SegformerSelfOutput(
                    (dense): Linear(in_features=256, out_features=256, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                )
                (drop_path): SegformerDropPath(p=0.08571428805589676)
                (layer_norm_2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (mlp): SegformerMixFFN(
                  (dense1): Linear(in_features=256, out_features=1024, bias=True)
                  (dwconv): SegformerDWConv(
                    (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                  )
                  (intermediate_act_fn): GELUActivation()
                  (dense2): Linear(in_features=1024, out_features=256, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (1): SegformerLayer(
                (layer_norm_1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (attention): SegformerAttention(
                  (self): SegformerEfficientSelfAttention(
                    (query): lora.Linear(
                      (base_layer): Linear(in_features=256, out_features=256, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=256, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=256, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (key): Linear(in_features=256, out_features=256, bias=True)
                    (value): lora.Linear(
                      (base_layer): Linear(in_features=256, out_features=256, bias=True)
                      (lora_dropout): ModuleDict(
                        (default): Dropout(p=0.1, inplace=False)
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=256, out_features=32, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=32, out_features=256, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
                      (lora_embedding_B): ParameterDict()
                    )
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                  (output): SegformerSelfOutput(
                    (dense): Linear(in_features=256, out_features=256, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                )
                (drop_path): SegformerDropPath(p=0.10000000149011612)
                (layer_norm_2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (mlp): SegformerMixFFN(
                  (dense1): Linear(in_features=256, out_features=1024, bias=True)
                  (dwconv): SegformerDWConv(
                    (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                  )
                  (intermediate_act_fn): GELUActivation()
                  (dense2): Linear(in_features=1024, out_features=256, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
            )
          )
          (layer_norm): ModuleList(
            (0): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
            (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (2): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
            (3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (decode_head): ModulesToSaveWrapper(
        (original_module): SegformerDecodeHead(
          (linear_c): ModuleList(
            (0): SegformerMLP(
              (proj): Linear(in_features=32, out_features=256, bias=True)
            )
            (1): SegformerMLP(
              (proj): Linear(in_features=64, out_features=256, bias=True)
            )
            (2): SegformerMLP(
              (proj): Linear(in_features=160, out_features=256, bias=True)
            )
            (3): SegformerMLP(
              (proj): Linear(in_features=256, out_features=256, bias=True)
            )
          )
          (linear_fuse): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): ReLU()
          (dropout): Dropout(p=0.1, inplace=False)
          (classifier): Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
        )
        (modules_to_save): ModuleDict(
          (default): SegformerDecodeHead(
            (linear_c): ModuleList(
              (0): SegformerMLP(
                (proj): Linear(in_features=32, out_features=256, bias=True)
              )
              (1): SegformerMLP(
                (proj): Linear(in_features=64, out_features=256, bias=True)
              )
              (2): SegformerMLP(
                (proj): Linear(in_features=160, out_features=256, bias=True)
              )
              (3): SegformerMLP(
                (proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (linear_fuse): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (batch_norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU()
            (dropout): Dropout(p=0.1, inplace=False)
            (classifier): Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
    )
  )
)
```

- [Wandb Report Link](https://api.wandb.ai/links/bow1226/hlvvkee3)
- [kaggle notebook link](https://www.kaggle.com/code/hsinwenchang/lightweightfinetunecomputervision/notebook)

- Inference Result:
![][image1]
- [Saved PEFT Model directory](https://github.com/polarbeargo/Apply-Lightweight-Fine-Tuning-LLMs/tree/main/segformer-scene-parse-150-lora)

## Future Improvements

- Increase the number of epochs for both notebook.
- For NLP notebook use a larger model such as [abacusai/Smaug-72B-v0.1](https://huggingface.co/abacusai/Smaug-72B-v0.1) to see if the performance improves if our hardware have enough memory to load the model.
- Increase the number of training data for both notebook.
- Try a larger SegFormer model variant such as [nvidia/segformer...](https://huggingface.co/models?search=segformer) to see if the performance improves if our hardware have enough memory to load the model.
- Perform hyperparameter tuning for both notebooks.
- Experiment with different values for the arguments that LoraConfig provides.
