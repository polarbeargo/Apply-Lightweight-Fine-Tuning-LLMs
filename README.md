# Lightweight Fine-Tuning Project
<!-- embedd output.png and output.png -->
[image1]: ./output.png
[image2]: ./images/gpt2train.png
[image3]: ./images/gpt2RunHistory.png
[image4]: ./images/GPT2Summary.png
[image5]: ./images/pefttrain.png
[image6]: ./images/peftRunHistory.png
[image7]: ./images/PEFTSummary.png

## Environment Setup

- Run the following commands to setup the environment:

```
chmod +x setupLightweightFineTuningLLMs.sh  
./setupLightweightFineTuningLLMs.sh
```

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

- [Wandb Report Link](https://wandb.ai/peft/peft/runs/3z5z3z3z?workspace=user-peft): To be updated soon.

- [kaggle notebook link](https://www.kaggle.com/peft/peft-gpt2): To be updated soon.

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

- [Wandb Report Link](https://api.wandb.ai/links/bow1226/r6g4akdw): To be updated soon.
- [kaggle notebook link](https://www.kaggle.com/peft/peft-qlora): To be updated soon.

- Inference Results: To be updated soon.

- [Saved PEFT Model directory](https://github.com/polarbeargo/Apply-Lightweight-Fine-Tuning-LLMs/tree/main/segformer-scene-parse-150-lora): To be updated soon.

#### nvidia/mit-b0 Model Architecture

```
```

#### After PEFT QLoRA nvidia/mit-b0 Model Architecture

```
```

- [Wandb Report Link](https://wandb.ai/peft/peft/runs/3z5z3z3z?workspace=user-peft): To be updated soon.
- [kaggle notebook link](https://www.kaggle.com/peft/peft-qlora): To be updated soon.

- Inference Result:
![][image1]
- [Saved PEFT Model directory](https://github.com/polarbeargo/Apply-Lightweight-Fine-Tuning-LLMs/tree/main/segformer-scene-parse-150-lora)

## Future Improvements

-
-
-
-
