# DENRL
Distantly-Supervised Joint Entity and Relation Extraction with Noise-Robust Learning ([PDF](https://arxiv.org/pdf/2310.04994.pdf))

<p align="center">
  <img src="figure/DENRL_architecture.png" width="80%" height="80%">
</p>


### Quick Start

- Python 3.8+
- Install requirements: `pip install -r requirements.txt`


- Train and evaluate a joint extraction model with noise reduction training (with instance selection)
```
MODEL_PATH=gpt2-medium
TRAIN_FILE="path/to/your/data"
VAL_FILE="path/to/your/data"
OUTPUT_DIR="results"

python run_jointmodel.py \
    --model_name_or_path $MODEL_PATH --classifier_type "crf" \
    --train_file $TRAIN_FILE --validation_file $VAL_FILE \
    --output_dir $OUTPUT_DIR --do_eval --do_train \
    --evaluation_strategy epoch --load_best_model_at_end \
    --metric_for_best_model eval_f1 --greater_is_better True \
    --per_device_train_batch_size 5 --per_device_eval_batch_size 20 \
    --gradient_accumulation_steps 16 --overwrite_cache \
    --use_negative_sampling --sample_rate 0.1 --num_train_epochs 100 \
    --beta 1.0 --alpha 0.5 --boot_start_epoch 5 --threshold 0.5 
``` 

- Train and evaluate a joint extraction model with standard training (without instance selection)
```
MODEL_PATH=gpt2-medium
TRAIN_FILE="path/to/your/data"
VAL_FILE="path/to/your/data"
OUTPUT_DIR="results"

python run_jointmodel.py \
    --model_name_or_path $MODEL_PATH --classifier_type "crf" \
    --train_file $TRAIN_FILE --validation_file $VAL_FILE \
    --output_dir $OUTPUT_DIR --do_eval --do_train \
    --evaluation_strategy epoch --load_best_model_at_end \
    --metric_for_best_model eval_f1 --greater_is_better True \
    --per_device_train_batch_size 5 --per_device_eval_batch_size 20 \
    --gradient_accumulation_steps 16 --overwrite_cache \
    --use_negative_sampling --sample_rate 0.1 --num_train_epochs 100 \
    --beta 1.0 --alpha 0.5 --boot_start_epoch 5 --threshold 0.5 --baseline
```

### Citation
If you find this repo useful, please cite our paper:
```bibtex
@article{li2023distantly,
  title={Distantly-Supervised Joint Entity and Relation Extraction with Noise-Robust Learning},
  author={Li, Yufei and Yu, Xiao and Guo, Yanghong and Liu, Yanchi and Chen, Haifeng and Liu, Cong},
  journal={arXiv preprint arXiv:2310.04994},
  year={2023}
}
```
