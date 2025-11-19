# MechanicsDSL

**A Domain-Specific Language for Classical Mechanics Simulation and Analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org)

MechanicsDSL is a comprehensive framework for symbolic and numerical analysis of classical mechanical systems. It bridges the gap between intuitive LaTeX-inspired notation and high-performance numerical simulation, enabling researchers and students to rapidly prototype and analyze complex dynamical systems.

## Features

- **Intuitive LaTeX-inspired syntax** for defining mechanical systems
- **Automatic equation derivation** using Euler-Lagrange formulation
- **Symbolic mathematics** with SymPy integration
- **High-performance numerical integration** via SciPy
- **3D visualization and animation** with energy analysis
- **Unit system** with dimensional analysis
- **Built-in validation** against analytical solutions
- **Export capabilities** for animations (MP4, GIF)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MechanicsDSL.git
cd MechanicsDSL

# Install dependencies
pip install -r requirements.txt
```

### Your First Simulation

```python
from mechanics_dsl import PhysicsCompiler

# Define a simple pendulum in DSL
dsl_code = r"""
\system{simple_pendulum}

\defvar{theta}{Angle}{rad}
\defvar{m}{Mass}{kg}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{\frac{1}{2} m l^2 \dot{theta}^2 - m g l (1 - \cos{\theta})}

\initial{theta=0.5, theta_dot=0.0}
"""

# Compile and simulate
compiler = PhysicsCompiler()
result = compiler.compile_dsl(dsl_code)
solution = compiler.simulate(t_span=(0, 10), num_points=1000)

# Visualize
compiler.animate(solution)
compiler.plot_energy(solution)
compiler.plot_phase_space(solution)
```

### Run Built-in Examples

```bash
# Simple pendulum
python mechanics_dsl.py --example simple_pendulum --energy --phase

# Double pendulum (chaos!)
python mechanics_dsl.py --example double_pendulum --time 30 --export chaos.mp4

# Harmonic oscillator
python mechanics_dsl.py --example harmonic_oscillator --validate
```

## DSL Syntax Reference

### System Definition

```latex
\system{system_name}
```

### Variable Declaration

```latex
\defvar{name}{type}{unit}
```

Types: `Angle`, `Position`, `Mass`, `Length`, `Velocity`, `Force`, etc.

### Parameter Definition

```latex
\parameter{name}{value}{unit}
```

### Lagrangian

```latex
\lagrangian{T - V}
```

Where `T` is kinetic energy and `V` is potential energy.

### Initial Conditions

```latex
\initial{q1=value1, q1_dot=value2, ...}
```

### Time Derivatives

```latex
\dot{x}      % First derivative
\ddot{x}     % Second derivative
```

### Mathematical Functions

```latex
\sin{expr}   \cos{expr}   \tan{expr}
\exp{expr}   \log{expr}   \sqrt{expr}
\frac{num}{denom}
```

## Examples

### Simple Pendulum

Derives the equation of motion:

```
θ̈ = -(g/l) sin(θ)
```

### Double Pendulum

Automatically handles the coupled nonlinear equations demonstrating chaotic behavior.

### Harmonic Oscillator

Validates against analytical solution: `x(t) = A cos(ωt + φ)`

## Architecture

```
DSL Source Code
      ↓
   Tokenizer  ──→  [Tokens]
      ↓
    Parser   ──→  [AST]
      ↓
Semantic Analyzer ──→ [System Representation]
      ↓
Symbolic Engine (SymPy) ──→ [Equations of Motion]
      ↓
Numerical Compiler ──→ [Optimized Functions]
      ↓
ODE Solver (SciPy) ──→ [Trajectory Data]
      ↓
Visualizer (Matplotlib) ──→ [Animations & Plots]
```

## Technical Details

### Symbolic Differentiation

The Euler-Lagrange equation is computed symbolically:

```
d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ = 0
```

### Numerical Integration

Multiple integration methods supported:
- **RK45** (default): Explicit Runge-Kutta 4(5)
- **DOP853**: High-order explicit method
- **LSODA**: Automatic stiffness detection
- **Radau**: Implicit method for stiff systems

### Energy Conservation Validation

Automatic verification of energy conservation for Hamiltonian systems with configurable tolerance.

## API Reference

### PhysicsCompiler

Main interface for the DSL system.

```python
compiler = PhysicsCompiler()

# Compile DSL code
result = compiler.compile_dsl(dsl_source)

# Run simulation
solution = compiler.simulate(
    t_span=(0, 10),      # Time interval
    num_points=1000,      # Output points
    method='RK45',        # Integration method
    rtol=1e-6,           # Relative tolerance
    atol=1e-8            # Absolute tolerance
)

# Visualize
compiler.animate(solution)
compiler.plot_energy(solution)
compiler.plot_phase_space(solution)
compiler.export_animation(solution, 'output.mp4')

# Get system information
info = compiler.get_info()
compiler.print_equations()
```

### SymbolicEngine

Low-level symbolic mathematics interface.

```python
from mechanics_dsl import SymbolicEngine

engine = SymbolicEngine()
symbol = engine.get_symbol('x')
expr = engine.ast_to_sympy(ast_node)
equations = engine.derive_equations_of_motion(lagrangian, coordinates)
```

### SystemValidator

Validation utilities for verifying correctness.

```python
from mechanics_dsl import SystemValidator

validator = SystemValidator()
passed = validator.validate_simple_harmonic_oscillator(compiler, solution)
passed = validator.validate_energy_conservation(compiler, solution)
```

## Performance

Benchmark results on Intel Core i7 (single-threaded):

| System | Compilation | Simulation (10s) | Total |
|--------|-------------|------------------|-------|
| Simple Pendulum | 0.05s | 0.12s | 0.17s |
| Double Pendulum | 0.08s | 0.45s | 0.53s |
| Harmonic Oscillator | 0.04s | 0.08s | 0.12s |

## Limitations

- Currently supports only Lagrangian formulation (Hamiltonian in development)
- Constraints must be holonomic
- Limited to classical (non-relativistic) mechanics
- 3D visualization optimized for pendulum-like systems

## Roadmap

- [ ] Hamiltonian formulation support
- [ ] Non-holonomic constraints
- [ ] Multi-body rigid dynamics
- [ ] Field theory support (waves, heat)
- [ ] GPU acceleration for large systems
- [ ] Interactive web interface
- [ ] Jupyter notebook integration
- [ ] Extension system for custom forces

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/yourusername/MechanicsDSL.git
cd MechanicsDSL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linter
flake8 mechanics_dsl/
```

## Citation

If you use MechanicsDSL in your research, please cite:

```bibtex
@article{parsons2025mechanicsdsl,
  title={MechanicsDSL: A Domain-Specific Language for Classical Mechanics Simulation},
  author={Parsons, Noah
  journal={Journal of Open Source Software},
  year={2025},
  note={In preparation}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Noah Parsons**
- Email: nomapa223@gmail.com
- GitHub: [@GuiloScion](https://github.com/GuiloScion)
- ORCID: [0009-0000-7224-6040](https://orcid.org/0009-0000-7224-6040)

## Related Projects

- [SymPy](https://www.sympy.org/) - Symbolic mathematics in Python
- [SciPy](https://scipy.org/) - Scientific computing library
- [Jupyter](https://jupyter.org/) - Interactive notebooks for computation

---

**Made with ❤️ for physics education and research**
