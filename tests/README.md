# Tests

## Running Tests

To run the tests, make sure you have the required dependencies installed:

```bash
pip install pytest pytest-asyncio pandas numpy scikit-learn
```

Then run the tests:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_ceteris_paribus_tool.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src/tools --cov-report=html
```

## Test Structure

- `tests/unit/` - Unit tests for individual components
  - `test_ceteris_paribus_tool.py` - Tests for the CeterisParibusTool

## Test Coverage

The `test_ceteris_paribus_tool.py` file includes:

1. **Initialization Tests**: Verify tool setup and explainable features detection
2. **Run Method Tests**: Test the `_run` method with various inputs
3. **Async Tests**: Test the `_arun` async method
4. **Integration Tests**: Test with realistic data patterns
5. **Error Handling**: Test error cases and edge cases

