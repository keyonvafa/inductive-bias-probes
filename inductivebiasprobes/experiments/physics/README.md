# Physics
## Data Generation

```
python generate_data.py
```

## Inductive Bias Test

### Generate white noise dataset

```
python generate_white_noise_and_oracle_predictions.py
```

### Fine-tune on white noise dataset

```
python train_model.py --config white_noise_config --pretrained next_token --model_type gpt --max_iters 100 --no_compile
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
