# Contributing to Rust HFT Arbitrage Lab

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rust-arblab.git
   cd rust-arblab
   ```
3. **Set up** the development environment (see README.md)
4. **Create** a branch for your changes:
   ```bash
   git checkout -b feature/my-new-feature
   ```

## 📝 Development Workflow

### Before You Start

- Check existing [issues](https://github.com/YOUR_USERNAME/rust-arblab/issues) and [pull requests](https://github.com/YOUR_USERNAME/rust-arblab/pulls)
- Open an issue to discuss major changes before implementing them
- Keep changes focused - one feature or fix per PR

### Making Changes

1. **Write clean code** following project conventions
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Test your changes** thoroughly

### Code Style

#### Python
- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://github.com/psf/black) for formatting:
  ```bash
  black python/ tests/
  ```
- Add type hints where appropriate
- Write docstrings for public functions

#### Rust
- Follow official [Rust style guidelines](https://doc.rust-lang.org/style-guide/)
- Use `cargo fmt` for formatting:
  ```bash
  cargo fmt
  ```
- Use `cargo clippy` for linting:
  ```bash
  cargo clippy -- -D warnings
  ```
- Add documentation comments (`///`) for public APIs

### Testing

Run the test suite before submitting:

```bash
# Python tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=python --cov-report=html

# Rust tests
cargo test

# Test Rust-Python bindings
maturin develop --release
pytest tests/test_rust_analytics.py
```

### Documentation

- Update `docs/` for new features or API changes
- Add examples for new functionality
- Keep README.md up to date
- Use clear, concise language

## 🔧 Pull Request Process

1. **Update** documentation and tests
2. **Ensure** all tests pass
3. **Commit** with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```
4. **Push** to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```
5. **Create** a Pull Request on GitHub
6. **Address** review feedback promptly

### PR Guidelines

- Title should be clear and descriptive
- Description should explain:
  - What changes were made
  - Why the changes were necessary
  - How to test the changes
- Link related issues using keywords (e.g., "Fixes #123")
- Keep PRs focused and reasonably sized
- Respond to review comments constructively

## 🐛 Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Detailed steps to reproduce the bug
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:
   - OS and version
   - Python version
   - Rust version
   - Relevant package versions
6. **Logs/Screenshots**: If applicable

Use the issue template if available.

## 💡 Suggesting Features

When suggesting features:

1. **Check** if the feature has already been requested
2. **Explain** the use case and benefits
3. **Describe** the proposed solution
4. **Consider** implementation complexity
5. **Be open** to discussion and alternatives

## 📋 Code Review

All submissions require review. We review PRs for:

- **Functionality**: Does it work as intended?
- **Code quality**: Is it clean, readable, and maintainable?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it properly documented?
- **Performance**: Are there any performance concerns?
- **Security**: Are there any security issues?

## 🏗️ Project Structure

```
rust-arblab/
├── app/                    # Streamlit dashboard
├── python/                 # Python implementations
├── rust_core/             # Rust core library
├── rust_python_bindings/  # PyO3 bindings
├── rust_connector/        # Exchange connectors
├── tests/                 # Test suite
├── examples/              # Jupyter notebooks
└── docs/                  # Documentation
```

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### High Priority
- Additional trading strategies
- Performance optimizations
- Test coverage improvements
- Documentation enhancements
- Bug fixes

### Medium Priority
- New exchange connectors
- Additional risk metrics
- UI/UX improvements
- Example notebooks

### Low Priority
- Code refactoring
- Style improvements
- Minor feature additions

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions
- Keep discussions professional

## 📞 Getting Help

- **Documentation**: Check [`docs/`](docs/) directory
- **Issues**: Search or create a new issue
- **Discussions**: Use GitHub Discussions for questions

## 🙏 Recognition

Contributors will be recognized in:
- Project README
- Release notes
- Contributor list

Thank you for contributing to Rust HFT Arbitrage Lab! 🚀
