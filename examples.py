# Generate example animations
from mechanics_dsl import run_example

# Simple pendulum
results = run_example('simple_pendulum', t_span=(0, 10))
results['compiler'].export_animation(results['solution'], 'images/pendulum.gif', fps=30)

# Double pendulum
results = run_example('double_pendulum', t_span=(0, 30))
results['compiler'].export_animation(results['solution'], 'images/double_pendulum_chaos.gif', fps=30)
