# Contributing to Vortex AI

Thank you for your interest in contributing to Vortex AI! This document outlines guidelines and development practices for contributing code, testing, or reporting issues.

## 🚀 Getting Started

1. **Fork the Repository**: Create a personal fork on GitHub.
2. **Clone the Fork**:
   ```bash
   git clone https://github.com/<your-username>/personal-data-assistant.git
   cd personal-data-assistant
   ```
3. **Configure Environment**:
   - Set up your `.env` file using `.env.example`.
   - Install dependencies in a virtual environment.
4. **Install Dev Quality Tools**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Code Style & Formatting

We use **Ruff** for linting and formatting. Before submitting any pull request, verify that your code conforms to the standard style guidelines:

- **Lint Check**:
  ```bash
  python -m ruff check .
  ```
- **Auto-Formatting**:
  ```bash
  python -m ruff format .
  ```

## 🧪 Testing

We require all contributions to pass the test suite. If you add new endpoints or modify the RAG pipeline, include corresponding tests in `tests/`:

- **Run Test Suite**:
  ```bash
  pytest tests/ -v
  ```
- **Check Coverage**:
  ```bash
  pytest --cov=app
  ```

## 📬 Submitting a Pull Request

1. Create a descriptive feature branch (`git checkout -b feature/your-feature`).
2. Commit your changes with clear, semantic messages.
3. Push to your fork and submit a Pull Request to the `main` branch.
4. Verify that the GitHub Actions CI workflow passes successfully on your PR.
