# Benchmarks, Algorithms, and Metrics for Hierarchical Disentanglement

This directory contains Python code to generate the Spaceshapes and Chopsticks benchmarks (`spaceshapes.py` and `chopsticks.py`), run the MIMOSA and COFHAE algorithms (`mimosa.py` and `cofhae.py`), and evaluate the `R^4` and `R^4_c` disentanglement metrics (`metrics.py`, with baselines adapted from [`disentanglement_lib`](https://github.com/google-research/disentanglement_lib)).

To run both COFHAE and MIMOSA together, run `main.py`, e.g. as follows:

```bash
python main.py --dataset=spaceshapes --output_dir=./foo --initial_dim=7
python main.py --dataset=chopsticks_depth2_either --output_dir=./bar --cos_simil_thresh=0.975
```

Note that the first time you run `main.py` for a particular dataset, it will be generated and cached in the `data/` folder. If you do not provide an `--output_dir`, it will be saved in a timestamped folder in `/tmp`.

## Main experiments in the paper

The following commands replicate one restart of our main results on all datasets:

```bash
for tau in 0.5 0.67 1.0
do
  for lmb1 in 10 100 1000
  do
    for lmb2 in 1 10 100
    do
      python main.py --dataset=chopsticks_depth2_slope --initial_dim=3 \
                     --softmax_temperature=$tau --assignment_penalty=$lmb1 --adversarial_penalty=$lmb2

      python main.py --dataset=chopsticks_depth2_inter --initial_dim=3 \
                     --softmax_temperature=$tau --assignment_penalty=$lmb1 --adversarial_penalty=$lmb2

      python main.py --dataset=chopsticks_depth2_either --initial_dim=4 \
                     --softmax_temperature=$tau --assignment_penalty=$lmb1 --adversarial_penalty=$lmb2

      python main.py --dataset=chopsticks_depth2_both --initial_dim=5 \
                     --softmax_temperature=$tau --assignment_penalty=$lmb1 --adversarial_penalty=$lmb2

      python main.py --dataset=chopsticks_depth3_slope --initial_dim=4 \
                     --softmax_temperature=$tau --assignment_penalty=$lmb1 --adversarial_penalty=$lmb2

      python main.py --dataset=chopsticks_depth3_inter --initial_dim=4 \
                     --softmax_temperature=$tau --assignment_penalty=$lmb1 --adversarial_penalty=$lmb2

      python main.py --dataset=chopsticks_depth3_either --initial_dim=5 \
                     --softmax_temperature=$tau --assignment_penalty=$lmb1 --adversarial_penalty=$lmb2

      python main.py --dataset=chopsticks_depth3_both --initial_dim=7 \
                     --softmax_temperature=$tau --assignment_penalty=$lmb1 --adversarial_penalty=$lmb2

      python main.py --dataset=spaceshapes --initial_dim=7 \
                     --cos_simil_thresh=0.95 --contagion_num=3 \
                     --softmax_temperature=$tau --assignment_penalty=$lmb1 --adversarial_penalty=$lmb2
    done
  done
done
```

Our result plotting code (not included here) then selects the best hyperparameter setting based on `train_mse` and `train_assign_err`.
