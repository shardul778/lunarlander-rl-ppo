# LunarLander Reinforcement Learning

A reinforcement-learning project using Gymnasium's `LunarLander-v3` environment and Proximal Policy Optimization (PPO). I used it to train a lander to control its thrust and rotation, compare reward-shaping ideas, and check how consistently saved models can land near the pad.

This is a simulation project; it does not control a physical rocket.

## What the project includes

- PPO training with Stable-Baselines3.
- Experiments with the standard environment reward and two custom reward wrappers: `HoverBonus` and `ProgressShaping`.
- Checkpoint saving and selection of a model based on evaluation episodes.
- Evaluation using episode reward, episode length, and a project-defined landing-precision check.
- Reward-progress and model-comparison plots.
- Video recordings of selected model runs.

## Results

The best saved PPO checkpoint in the recorded comparison was evaluated over 50 episodes. The same set of evaluation seeds was used to compare the candidate checkpoints, and the best checkpoint was selected using these results.

| Metric | Observed result |
|---|---:|
| Mean episode reward | 221.47 |
| Reward standard deviation | 37.20 |
| Episodes with reward of at least 200 | 44 / 50 (88%) |
| Episodes meeting the project's precision criteria | 39 / 50 (78%) |
| Timesteps on selected checkpoint | 255,960 |

The selected model's two recorded demonstration episodes scored 225.6 and 251.0. These two episodes are examples, not a separate performance estimate.

### How “precise landing” is measured

For this project, an episode counts as a precise landing only if all of these conditions are met:

- The episode terminates normally rather than timing out.
- Total episode reward is at least 100.
- Both landing legs register contact.
- Absolute horizontal position is at most `0.15` from the pad centre.
- Absolute horizontal velocity and vertical velocity are each at most `0.15`.
- Absolute landing angle is at most `0.15` radians.

These thresholds are custom project criteria, not an official Gymnasium metric. The reported 78% was measured on the candidate-comparison episodes used to select the model, so it should not be presented as an independent test score. Testing on fresh seeds is needed for a less biased estimate.

## Tech stack

- Python
- Gymnasium / Box2D
- Stable-Baselines3 (PPO)
- NumPy and Pandas
- Matplotlib
- MoviePy / Gymnasium `RecordVideo`
- Google Colab for training and evaluation

## Run it

The experiments were developed in Google Colab. To reproduce them, upload the notebook from this repository to Colab and run the setup cell first.

Install the dependencies in a Python environment:

```bash
pip install -r requirements.txt
```

The initial comparison script trains multiple reward-shaping experiments and can take a while to run. It is not just a quick inference script. To view the selected model without retraining, load `models/final_selected_model.zip` in the notebook's evaluation/video cells.

## Repository layout

After organising the files for GitHub, the repository should look like this:

```text
lunarlander-rl/
├── README.md
├── requirements.txt
├── .gitignore
├── train_compare.py
├── notebooks/
│   └── lunarlander_rl.ipynb
├── models/
│   └── final_selected_model.zip
├── results/
│   ├── reward_shaping_results.json
│   ├── candidate_comparison_50_episodes.csv
│   ├── learning_curve_300k.png
│   └── candidate_comparison_300k.png
└── videos/
    └── landing_demo.mp4
```

The notebook, checkpoint, plots, result files, and video should be copied from the corresponding locations in the Colab workspace. The structure above is the intended GitHub layout, not a claim that those files are already present in this starter folder.

## Limitations

- LunarLander is a simplified two-dimensional environment, so results do not transfer directly to real flight control.
- PPO training is stochastic; performance can vary between random seeds.
- Reward of 200 is a commonly used benchmark threshold, but it does not by itself prove a centred, gentle landing.
- The custom precision criteria are stricter than a reward-only check and should be evaluated on fresh seeds before making broad claims.

## Next steps

- Test the selected model on fresh random seeds and report the results separately from model selection.
- Compare performance across multiple independent PPO training seeds.
- Add an automated evaluation command that produces the metrics and plots from the saved checkpoint.

## References

- [Gymnasium LunarLander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/)
