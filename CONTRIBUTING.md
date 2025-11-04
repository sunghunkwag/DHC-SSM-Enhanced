# Contributing to DHC-SSM

This document provides guidelines and instructions for contributing to the DHC-SSM Architecture project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct. All participants are expected to uphold professional and respectful behavior in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/dhc-ssm-v3.git
   cd dhc-ssm-v3
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/originalowner/dhc-ssm-v3.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- PyTorch 2.9.0 or higher
- Git

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Verify the bug has not already been reported
- Include:
  - Python version
  - PyTorch version
  - Operating system
  - Minimal code to reproduce
  - Expected versus actual behavior
  - Error messages and stack traces

### Suggesting Enhancements

- Use the GitHub issue tracker
- Provide clear description of the enhancement
- Explain the rationale and use cases
- Include examples where applicable

### Contributing Code

1. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes:
   - Write clean, readable code
   - Follow coding standards
   - Add tests for new features
   - Update documentation

3. Test your changes:
   ```bash
   pytest tests/
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request on GitHub

## Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use Black for code formatting:
  ```bash
  black dhc_ssm/
  ```
- Use isort for import sorting:
  ```bash
  isort dhc_ssm/
  ```
- Use flake8 for linting:
  ```bash
  flake8 dhc_ssm/
  ```

### Type Hints

- Add type hints to all functions:
  ```python
  def process_data(x: torch.Tensor, config: DHCSSMConfig) -> torch.Tensor:
      ...
  ```
- Use mypy for type checking:
  ```bash
  mypy dhc_ssm/
  ```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
    """
    ...
```

### Code Organization

- Keep functions focused and concise
- Use meaningful variable names
- Add comments for complex logic
- Avoid deep nesting (maximum 3-4 levels)
- Extract magic numbers to constants

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for setup
- Target high code coverage (>90%)

Example test:

```python
def test_forward_pass():
    """Test forward pass with various inputs."""
    config = get_debug_config()
    model = DHCSSMModel(config)
    
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    
    assert output.shape == (4, config.output_dim)
    assert not torch.isnan(output).any()
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest tests/ --cov=dhc_ssm --cov-report=html

# Run with verbose output
pytest tests/ -v
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include usage examples in docstrings
- Keep documentation synchronized with code changes

### README and Guides

- Update README.md for user-facing changes
- Add examples for new features
- Update CHANGELOG.md

### API Documentation

- Document all public APIs
- Include parameter types and return types
- Provide usage examples

## Pull Request Process

### Before Submitting

1. Update your branch with latest upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Run all tests:
   ```bash
   pytest tests/
   ```

3. Check code style:
   ```bash
   black --check dhc_ssm/
   flake8 dhc_ssm/
   mypy dhc_ssm/
   ```

4. Update documentation if needed

### PR Description

Include in your PR description:

- What: Brief description of changes
- Why: Motivation for the changes
- How: Technical details if complex
- Testing: How you tested the changes
- Related Issues: Link to related issues

Example:

```markdown
## What
Add support for mixed precision training

## Why
Improves training speed and reduces memory usage

## How
- Added AMP context manager to training loop
- Updated loss scaling logic
- Added configuration option

## Testing
- Tested with debug and small configs
- Verified memory reduction (~20%)
- Confirmed accuracy maintained

## Related Issues
Closes #123
```

### Review Process

- Maintainers will review your PR
- Address review comments
- Update your PR as needed
- Merge occurs after approval

## Development Workflow

### Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Urgent fixes

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(model): add mixed precision training support

fix(trainer): resolve gradient accumulation bug

docs(readme): update installation instructions
```

## Questions

- Open an issue for questions
- Join discussions on GitHub Discussions
- Review existing issues and PRs

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
