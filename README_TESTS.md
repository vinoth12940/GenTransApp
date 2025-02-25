# Format Preservation Tests

This directory contains tests to validate the format preservation capabilities of the Translation API across different providers, models, and text formatting styles.

## Test Overview

The tests validate the following aspects of format preservation:

1. **Multiple formatting types:**

   - Bullet lists and nested lists
   - Numbered lists
   - Tables
   - Headers and subheaders
   - Bold text formatting
   - Mixed formatting (combination of all types)
2. **All supported language pairs:**

   - English → Spanish
   - English → Chinese
   - Spanish → English
3. **Both domains:**

   - Healthcare
   - Insurance
4. **All supported providers and models:**

   - OpenAI (GPT-4o)
   - Anthropic (Claude 3.5 Sonnet)
   - Google (Gemini models)
   - Groq (Llama models)
   - Cohere, DeepSeek, and Hugging Face models

## Running the Tests

### Prerequisites

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Ensure that all API keys are properly set in your `.env` file.

### Run All Tests

To run all format preservation tests:

```bash
python test_format_preservation.py
```

This will execute the full test suite and generate the following JSON result files:

- `format_test_results.json`: Results for all models with different formatting
- `language_pair_test_results.json`: Results for all language pairs
- `domain_test_results.json`: Domain-specific formatting results
- `preserve_formatting_test_results.json`: Impact of the `preserve_formatting` flag

### Run with pytest

To run specific test cases:

```bash
# Run all tests
pytest -xvs test_format_preservation.py

# Run just one test
pytest -xvs test_format_preservation.py::test_all_formats_all_models
```

## Test Description

### test_all_formats_all_models

Tests each model with all formatting types (bullet lists, numbered lists, tables, mixed) to verify format preservation.

### test_mixed_formatting_languages

Tests mixed formatting (most complex case) with all language pairs for a specific model.

### test_domain_formatting

Tests domain-specific formatting differences across healthcare and insurance domains.

### test_preserve_formatting_flag

Compares formatting preservation with the `preserve_formatting` flag set to `True` vs `False`.

## Validation Criteria

The format preservation check validates:

1. Line count preservation (with small tolerance)
2. Bullet point count preservation
3. Numbered list item preservation
4. Table row count preservation
5. Header count preservation
6. Bold text format preservation

## Results Analysis

After running the tests, review the JSON result files to:

1. Identify which models perform best for format preservation
2. See which formatting elements are most reliably preserved
3. Compare performance across language pairs
4. Verify the impact of the `preserve_formatting` flag

The success criterion is a 70% overall pass rate for all tests combined.
