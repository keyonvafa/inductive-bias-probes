# Othello
## Data Generation

To generate the data for Othello experiments, you need to first download the `othello_synthetic` folder from [link to Google Drive](https://drive.google.com/drive/folders/1pDMdMrnxMRiDnUd-CNfRNvZCi7VXFRtv), from the [Othello World](https://github.com/likenneth/othello_world?tab=readme-ov-file) repo.
```py
python generate_data.py
```

## Pretrain

Pretrain next-token predictor

```py
python train_model.py --config ntp_config --max_iters 1_600_000  --batch_size 256 --eval_interval 1_000 --eval_iters 1 --model_type rnn
python train_model.py --config ntp_config --max_iters 1_600_000  --batch_size 256 --eval_interval 1_000 --eval_iters 1 --model_type lstm
python train_model.py --config ntp_config --max_iters 1_600_000  --batch_size 256 --eval_interval 1_000 --eval_iters 1 --model_type gpt
python train_model.py --config ntp_config --max_iters 1_600_000  --batch_size 256 --eval_interval 1_000 --eval_iters 1 --model_type mamba
python train_model.py --config ntp_config --max_iters 1_600_000  --batch_size 256 --eval_interval 1_000 --eval_iters 1 --model_type mamba2
```

## Pretrained Model Evaluation (Reproduce Table 6)
```py
python compute_legal_next_moves.py
```

## Transfer Fine-tuning

```py
python train_model.py --model_type rnn --config transfer_balance_black_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type rnn --config transfer_balance_black_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type rnn --config transfer_majority_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type rnn --config transfer_majority_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type rnn --config transfer_edges_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type rnn --config transfer_edges_config --pretrained next_token --max_iters 5_000

python train_model.py --model_type lstm --config transfer_balance_black_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type lstm --config transfer_balance_black_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type lstm --config transfer_majority_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type lstm --config transfer_majority_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type lstm --config transfer_edges_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type lstm --config transfer_edges_config --pretrained next_token --max_iters 5_000

python train_model.py --model_type gpt --config transfer_balance_black_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type gpt --config transfer_balance_black_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type gpt --config transfer_majority_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type gpt --config transfer_majority_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type gpt --config transfer_edges_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type gpt --config transfer_edges_config --pretrained next_token --max_iters 5_000

python train_model.py --model_type mamba --config transfer_balance_black_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type mamba --config transfer_balance_black_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type mamba --config transfer_majority_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type mamba --config transfer_majority_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type mamba --config transfer_edges_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type mamba --config transfer_edges_config --pretrained next_token --max_iters 5_000

python train_model.py --model_type mamba2 --config transfer_balance_black_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type mamba2 --config transfer_balance_black_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type mamba2 --config transfer_majority_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type mamba2 --config transfer_majority_config --pretrained next_token --max_iters 5_000
python train_model.py --model_type mamba2 --config transfer_edges_config --pretrained scratch --max_iters 5_000
python train_model.py --model_type mamba2 --config transfer_edges_config --pretrained next_token --max_iters 5_000
```

Transfer to learn board state (Reconstruct Figure 6)
```py
python train_model.py --model_type mamba --config state_config --pretrained next_token --max_iters 10_000 --reconstruct_board --reconstruction_eval_interval 1_000
python train_model.py --model_type mamba2 --config state_config --pretrained next_token --max_iters 10_000 --reconstruct_board --reconstruction_eval_interval 1_000
```

## Inductive Bias Test

### Generate white noise dataset
```py
python generate_white_noise.py
```

### Fine-tune on white noise dataset 
```py
python train_model.py --model_type gpt --config white_noise_config --pretrained next_token --white_noise_dataset_size 100 --max_iters 100
python train_model.py --model_type gpt --config white_noise_config --pretrained scratch --white_noise_dataset_size 100 --max_iters 100

python train_model.py --model_type rnn --config white_noise_config --pretrained next_token --white_noise_dataset_size 100 --max_iters 100
python train_model.py --model_type rnn --config white_noise_config --pretrained scratch --white_noise_dataset_size 100 --max_iters 100

python train_model.py --model_type lstm --config white_noise_config --pretrained next_token --white_noise_dataset_size 100 --max_iters 100
python train_model.py --model_type lstm --config white_noise_config --pretrained scratch --white_noise_dataset_size 100 --max_iters 100

python train_model.py --model_type mamba --config white_noise_config --pretrained next_token --white_noise_dataset_size 100 --max_iters 100
python train_model.py --model_type mamba --config white_noise_config --pretrained scratch --white_noise_dataset_size 100 --max_iters 100

python train_model.py --model_type mamba2 --config white_noise_config --pretrained next_token --white_noise_dataset_size 100 --max_iters 100
python train_model.py --model_type mamba2 --config white_noise_config --pretrained scratch --white_noise_dataset_size 100 --max_iters 100
```

### Compute inductive bias

```py
python compute_inductive_bias.py --model_type gpt --pretrained scratch
python compute_inductive_bias.py --model_type gpt --pretrained next_token
python compute_inductive_bias.py --model_type rnn --pretrained scratch
python compute_inductive_bias.py --model_type rnn --pretrained next_token
python compute_inductive_bias.py --model_type lstm --pretrained scratch
python compute_inductive_bias.py --model_type lstm --pretrained next_token
python compute_inductive_bias.py --model_type mamba --pretrained scratch
python compute_inductive_bias.py --model_type mamba --pretrained next_token
python compute_inductive_bias.py --model_type mamba2 --pretrained scratch
python compute_inductive_bias.py --model_type mamba2 --pretrained next_token
```

#### Compute transfer metrics (Reproduce Table 9)
```py
python compute_transfer_metrics.py
```

### Ablations
```py
python generate_white_noise.py --white_noise_dataset_size 10
python generate_white_noise.py --white_noise_dataset_size 50
python generate_white_noise.py --white_noise_dataset_size 100
python generate_white_noise.py --white_noise_dataset_size 500
```

Reproduce Table 4
```py
python compute_inductive_bias.py --max_iters 10 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --max_iters 10 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --max_iters 10 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --max_iters 10 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --max_iters 10 --model_type mamba2 --pretrained next_token

python compute_inductive_bias.py --max_iters 50 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --max_iters 50 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --max_iters 50 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --max_iters 50 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --max_iters 50 --model_type mamba2 --pretrained next_token

python compute_inductive_bias.py --max_iters 100 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --max_iters 100 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --max_iters 100 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --max_iters 100 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --max_iters 100 --model_type mamba2 --pretrained next_token

python compute_inductive_bias.py --max_iters 500 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --max_iters 500 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --max_iters 500 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --max_iters 500 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --max_iters 500 --model_type mamba2 --pretrained next_token
```

```py
python compute_inductive_bias.py --max_iters 10 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --max_iters 10 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --max_iters 10 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --max_iters 10 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --max_iters 10 --model_type mamba2 --pretrained next_token

python compute_inductive_bias.py --max_iters 50 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --max_iters 50 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --max_iters 50 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --max_iters 50 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --max_iters 50 --model_type mamba2 --pretrained next_token

python compute_inductive_bias.py --max_iters 100 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --max_iters 100 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --max_iters 100 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --max_iters 100 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --max_iters 100 --model_type mamba2 --pretrained next_token

python compute_inductive_bias.py --max_iters 500 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --max_iters 500 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --max_iters 500 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --max_iters 500 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --max_iters 500 --model_type mamba2 --pretrained next_token
```

Reproduce Table 5
```py
python train_model.py --model_type gpt --config white_noise_config --pretrained next_token --white_noise_dataset_size 10
python train_model.py --model_type rnn --config white_noise_config --pretrained next_token --white_noise_dataset_size 10
python train_model.py --model_type lstm --config white_noise_config --pretrained next_token --white_noise_dataset_size 10
python train_model.py --model_type mamba --config white_noise_config --pretrained next_token --white_noise_dataset_size 10
python train_model.py --model_type mamba2 --config white_noise_config --pretrained next_token --white_noise_dataset_size 10

python train_model.py --model_type gpt --config white_noise_config --pretrained next_token --white_noise_dataset_size 50
python train_model.py --model_type rnn --config white_noise_config --pretrained next_token --white_noise_dataset_size 50
python train_model.py --model_type lstm --config white_noise_config --pretrained next_token --white_noise_dataset_size 50
python train_model.py --model_type mamba --config white_noise_config --pretrained next_token --white_noise_dataset_size 50
python train_model.py --model_type mamba2 --config white_noise_config --pretrained next_token --white_noise_dataset_size 50

python train_model.py --model_type gpt --config white_noise_config --pretrained next_token --white_noise_dataset_size 100
python train_model.py --model_type rnn --config white_noise_config --pretrained next_token --white_noise_dataset_size 100
python train_model.py --model_type lstm --config white_noise_config --pretrained next_token --white_noise_dataset_size 100
python train_model.py --model_type mamba --config white_noise_config --pretrained next_token --white_noise_dataset_size 100
python train_model.py --model_type mamba2 --config white_noise_config --pretrained next_token --white_noise_dataset_size 100

python train_model.py --model_type gpt --config white_noise_config --pretrained next_token --white_noise_dataset_size 500
python train_model.py --model_type rnn --config white_noise_config --pretrained next_token --white_noise_dataset_size 500
python train_model.py --model_type lstm --config white_noise_config --pretrained next_token --white_noise_dataset_size 500
python train_model.py --model_type mamba --config white_noise_config --pretrained next_token --white_noise_dataset_size 500
python train_model.py --model_type mamba2 --config white_noise_config --pretrained next_token --white_noise_dataset_size 500
```

```py
python compute_inductive_bias.py --white_noise_dataset_size 10 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 10 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 10 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 10 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 10 --model_type mamba2 --pretrained next_token

python compute_inductive_bias.py --white_noise_dataset_size 50 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 50 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 50 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 50 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 50 --model_type mamba2 --pretrained next_token

python compute_inductive_bias.py --white_noise_dataset_size 100 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 100 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 100 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 100 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 100 --model_type mamba2 --pretrained next_token

python compute_inductive_bias.py --white_noise_dataset_size 500 --model_type rnn --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 500 --model_type lstm --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 500 --model_type gpt --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 500 --model_type mamba --pretrained next_token
python compute_inductive_bias.py --white_noise_dataset_size 500 --model_type mamba2 --pretrained next_token
```