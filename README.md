# Bayesian Neural Networks

Project for Bayesian Statistics Class.

In this repository, we implement two approaches for introducing uncertainty in neural networks based on the code available in tensorflow_probability examples.

# Get started

```frequentist.py``` implements a frequentist approach. It can be launched from the terminal by simply choosing the right environment containing the following packages: seaborn, numpy, tensorflow, pyplot. The command is the following:

```python frequentist.py --learning_rate --max_steps --batch_size --data_dir --model_dir --viz_steps --print_steps --run_steps```

Default values are those written in the associated report.


```bayesian.py``` implements a bayesian variational approach. It can be launched from the terminal by simply choosing the right environment containing the following packages: seaborn, numpy, tensorflow_probability, pyplot. The command is the following:

```python bayesian.py --learning_rate --max_steps --batch_size --data_dir --model_dir --viz_steps --print_steps --num_monte_carlo```

Default values are those written in the associated report.

WARNING
The ```--fake_data``` flag should always be ```True``` which is the default value. Program will not run otherwise.
