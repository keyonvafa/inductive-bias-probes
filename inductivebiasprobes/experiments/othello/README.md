```py
python generate_data.py
```

```py
torchrun --nproc_per_node=4 train_model.py --config ntp_config --gradient_accumulation_steps 4  --max_iters 400_000  --batch_size 256 --eval_interval 1_000 --eval_iters 1 --model_type mamba
```