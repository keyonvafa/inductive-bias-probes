# Lattice / Gridworld
## Data Generation

```
python generate_data.py
```

## Pretrain

Pretrain next-token predictor

```
python train_model.py --config ntp_config --num_states 2 --max_iters 10_000 --model_type gpt
python train_model.py --config ntp_config --num_states 3 --max_iters 10_000 --model_type gpt
python train_model.py --config ntp_config --num_states 4 --max_iters 10_000 --model_type gpt
python train_model.py --config ntp_config --num_states 5 --max_iters 10_000 --model_type gpt

python train_model.py --config ntp_config --num_states 2 --max_iters 10_000 --model_type mamba
python train_model.py --config ntp_config --num_states 3 --max_iters 10_000 --model_type mamba
python train_model.py --config ntp_config --num_states 4 --max_iters 10_000 --model_type mamba
python train_model.py --config ntp_config --num_states 5 --max_iters 10_000 --model_type mamba

python train_model.py --config ntp_config --num_states 2 --max_iters 10_000 --model_type mamba2
python train_model.py --config ntp_config --num_states 3 --max_iters 10_000 --model_type mamba2
python train_model.py --config ntp_config --num_states 4 --max_iters 10_000 --model_type mamba2
python train_model.py --config ntp_config --num_states 5 --max_iters 10_000 --model_type mamba2

python train_model.py --config ntp_config --num_states 2 --max_iters 10_000 --model_type rnn
python train_model.py --config ntp_config --num_states 3 --max_iters 10_000 --model_type rnn
python train_model.py --config ntp_config --num_states 4 --max_iters 10_000 --model_type rnn
python train_model.py --config ntp_config --num_states 5 --max_iters 10_000 --model_type rnn

python train_model.py --config ntp_config --num_states 2 --max_iters 10_000 --model_type lstm
python train_model.py --config ntp_config --num_states 3 --max_iters 10_000 --model_type lstm
python train_model.py --config ntp_config --num_states 4 --max_iters 10_000 --model_type lstm
python train_model.py --config ntp_config --num_states 5 --max_iters 10_000 --model_type lstm
```

## Inductive Bias Test

### Generate white noise dataset

```
python generate_white_noise.py
```

### Fine-tune on white noise dataset (for Figure 5)

```
python train_model.py --config white_noise_config --pretrained next_token --num_states 2 --model_type gpt --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 3 --model_type gpt --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 4 --model_type gpt --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type gpt --max_iters 100

python train_model.py --config white_noise_config --pretrained next_token --num_states 2 --model_type mamba --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 3 --model_type mamba --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 4 --model_type mamba --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type mamba --max_iters 100

python train_model.py --config white_noise_config --pretrained next_token --num_states 2 --model_type mamba2 --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 3 --model_type mamba2 --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 4 --model_type mamba2 --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type mamba2 --max_iters 100

python train_model.py --config white_noise_config --pretrained next_token --num_states 2 --model_type rnn --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 3 --model_type rnn --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 4 --model_type rnn --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type rnn --max_iters 100

python train_model.py --config white_noise_config --pretrained next_token --num_states 2 --model_type lstm --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 3 --model_type lstm --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 4 --model_type lstm --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type lstm --max_iters 100
```

### Fine-tune on white noise dataset (for Table 2)

```
python train_model.py --config white_noise_config --pretrained scratch --num_states 5 --model_type gpt --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type gpt --max_iters 100

python train_model.py --config white_noise_config --pretrained scratch --num_states 5 --model_type rnn --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type rnn --max_iters 100

python train_model.py --config white_noise_config --pretrained scratch --num_states 5 --model_type lstm --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type lstm --max_iters 100

python train_model.py --config white_noise_config --pretrained scratch --num_states 5 --model_type mamba --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type mamba --max_iters 100

python train_model.py --config white_noise_config --pretrained scratch --num_states 5 --model_type mamba2 --max_iters 100
python train_model.py --config white_noise_config --pretrained next_token --num_states 5 --model_type mamba2 --max_iters 100
```

### Compute inductive bias (for Figure 5)

```
python compute_inductive_bias.py --pretrained next_token --num_states 2 --model_type gpt
python compute_inductive_bias.py --pretrained next_token --num_states 2 --model_type mamba
python compute_inductive_bias.py --pretrained next_token --num_states 2 --model_type mamba2
python compute_inductive_bias.py --pretrained next_token --num_states 2 --model_type rnn
python compute_inductive_bias.py --pretrained next_token --num_states 2 --model_type lstm

python compute_inductive_bias.py --pretrained next_token --num_states 3 --model_type gpt
python compute_inductive_bias.py --pretrained next_token --num_states 3 --model_type mamba
python compute_inductive_bias.py --pretrained next_token --num_states 3 --model_type mamba2
python compute_inductive_bias.py --pretrained next_token --num_states 3 --model_type rnn
python compute_inductive_bias.py --pretrained next_token --num_states 3 --model_type lstm

python compute_inductive_bias.py --pretrained next_token --num_states 4 --model_type gpt
python compute_inductive_bias.py --pretrained next_token --num_states 4 --model_type mamba
python compute_inductive_bias.py --pretrained next_token --num_states 4 --model_type mamba2
python compute_inductive_bias.py --pretrained next_token --num_states 4 --model_type rnn
python compute_inductive_bias.py --pretrained next_token --num_states 4 --model_type lstm

python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type gpt
python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type mamba
python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type mamba2
python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type rnn
python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type lstm
```

### Compute inductive bias (for Table 2)

```
python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type gpt
python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type mamba
python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type mamba2
python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type rnn
python compute_inductive_bias.py --pretrained next_token --num_states 5 --model_type lstm

python compute_inductive_bias.py --pretrained scratch --num_states 5 --model_type gpt
python compute_inductive_bias.py --pretrained scratch --num_states 5 --model_type mamba
python compute_inductive_bias.py --pretrained scratch --num_states 5 --model_type mamba2
python compute_inductive_bias.py --pretrained scratch --num_states 5 --model_type rnn
python compute_inductive_bias.py --pretrained scratch --num_states 5 --model_type lstm
```

### Generate Figure 5
```
python plot_ib_num_states.py 
```
