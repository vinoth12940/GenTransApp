import pytest
import asyncio
import json
import os
from dotenv import load_dotenv
from app import UnifiedTranslator, LanguageCode, Provider, Model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize translator
translator = UnifiedTranslator()

# Test data with various formatting elements
TEST_CASES = {
    "bullet_lists": {
        "text": """
# Medical Report

## Patient Information
* **Name**: John Doe
* **DOB**: 01/15/1980
* **MRN**: 12345678

## Medical History
* **Hypertension**: Diagnosed in 2018
* **Type 2 Diabetes**: Diagnosed in 2019
  * HbA1c: 7.2% (last checked 2023-05-15)
  * Currently managed with Metformin 1000mg twice daily
* **Allergies**:
  * Penicillin - Severe rash
  * Shellfish - Anaphylaxis
""",
        "domain": "healthcare"
    },
    "numbered_lists": {
        "text": """
# Insurance Claim Process

Please follow these steps to file a claim:

1. Report the incident within 24 hours
2. Complete the claim form with the following details:
   a. Policy number
   b. Date and time of incident
   c. Detailed description of loss
3. Submit supporting documentation:
   - Photos of damage
   - Police report (if applicable)
   - Repair estimates
4. Sign and date the declaration
""",
        "domain": "insurance"
    },
    "tables": {
        "text": """
# Medication Schedule

| Medication | Dosage | Frequency | Purpose |
|------------|--------|-----------|---------|
| Lisinopril | 10mg | Once daily | Hypertension |
| Metformin | 1000mg | Twice daily | Diabetes |
| Aspirin | 81mg | Once daily | Cardiac prophylaxis |
| Atorvastatin | 20mg | Once daily | Cholesterol |
""",
        "domain": "healthcare"
    },
    "mixed_formatting": {
        "text": """
# Property Insurance Policy

## Coverage Summary

* **Building Coverage**: $250,000
* **Personal Property**: $125,000
* **Liability**: $300,000
* **Deductible**: $1,000

## Exclusions

This policy does NOT cover losses resulting from:

1. Flood damage
2. Earthquake damage
3. Intentional acts
4. War or nuclear hazard

## Premium Schedule

| Payment Option | Amount | Due Date |
|---------------|--------|----------|
| Annual | $1,200 | January 1 |
| Semi-annual | $625 | Jan 1 & Jul 1 |
| Quarterly | $325 | Jan/Apr/Jul/Oct 1 |
""",
        "domain": "insurance"
    }
}

# Language pairs to test
LANGUAGE_PAIRS = [
    (LanguageCode.ENGLISH, LanguageCode.SPANISH),
    (LanguageCode.ENGLISH, LanguageCode.CHINESE),
    (LanguageCode.SPANISH, LanguageCode.ENGLISH)
]

# Helper function to check format preservation
def check_format_preservation(source_text, translated_text):
    """
    Checks if key formatting elements are preserved between source and translation
    Returns (passed, details) tuple
    """
    checks = []
    
    # Check for equivalent line count (allowing for small differences)
    source_lines = source_text.strip().split('\n')
    translated_lines = translated_text.strip().split('\n')
    line_diff = abs(len(source_lines) - len(translated_lines))
    checks.append({
        "check": "Line count",
        "passed": line_diff <= 3,  # Allow small differences in line wrapping
        "details": f"Source: {len(source_lines)} lines, Translation: {len(translated_lines)} lines"
    })
    
    # Check for equivalent bullet point count
    source_bullets = sum(1 for line in source_lines if line.strip().startswith('*') or line.strip().startswith('-'))
    translated_bullets = sum(1 for line in translated_lines if line.strip().startswith('*') or line.strip().startswith('-'))
    checks.append({
        "check": "Bullet points",
        "passed": source_bullets == translated_bullets,
        "details": f"Source: {source_bullets} bullets, Translation: {translated_bullets} bullets"
    })
    
    # Check for equivalent numbered list items
    source_numbered = sum(1 for line in source_lines if line.strip() and line.strip()[0].isdigit() and '. ' in line)
    translated_numbered = sum(1 for line in translated_lines if line.strip() and line.strip()[0].isdigit() and '. ' in line)
    checks.append({
        "check": "Numbered items",
        "passed": source_numbered == translated_numbered,
        "details": f"Source: {source_numbered} numbered items, Translation: {translated_numbered} numbered items"
    })
    
    # Check for table row markers
    source_table_rows = sum(1 for line in source_lines if line.strip().startswith('|') and line.strip().endswith('|'))
    translated_table_rows = sum(1 for line in translated_lines if line.strip().startswith('|') and line.strip().endswith('|'))
    checks.append({
        "check": "Table rows",
        "passed": source_table_rows == translated_table_rows,
        "details": f"Source: {source_table_rows} table rows, Translation: {translated_table_rows} table rows"
    })
    
    # Check for headers (# markers)
    source_headers = sum(1 for line in source_lines if line.strip().startswith('#'))
    translated_headers = sum(1 for line in translated_lines if line.strip().startswith('#'))
    checks.append({
        "check": "Headers",
        "passed": source_headers == translated_headers,
        "details": f"Source: {source_headers} headers, Translation: {translated_headers} headers"
    })
    
    # Check for bold markers
    source_bold = sum(1 for line in source_lines if '**' in line)
    translated_bold = sum(1 for line in translated_lines if '**' in line)
    checks.append({
        "check": "Bold formatting",
        "passed": source_bold == translated_bold,
        "details": f"Source: {source_bold} bold elements, Translation: {translated_bold} bold elements"
    })
    
    # Overall check
    passed = all(check["passed"] for check in checks)
    
    return passed, checks

# Generate test functions for each model and test case
@pytest.mark.asyncio
async def test_all_formats_all_models():
    """Run comprehensive format preservation tests across models and formatting types"""
    
    results = {
        "tests": [],
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "models_tested": set()
        }
    }
    
    # Select models to test - one from each provider
    models_to_test = [
        (Provider.OPENAI, Model.GPT_4O),
        (Provider.ANTHROPIC, Model.CLAUDE_3_SONNET),
        (Provider.GOOGLE, Model.GEMINI_1_5_PRO),
        (Provider.GROQ, Model.LLAMA_3_70B),
        (Provider.COHERE, Model.COHERE_COMMAND_R_PLUS),
        (Provider.DEEPSEEK, Model.DEEPSEEK_CHAT),
        (Provider.HUGGINGFACE, Model.HUGGINGFACE_MISTRAL_7B)
    ]
    
    # Test each model with different formatting scenarios and language pairs
    for provider, model in models_to_test:
        provider_name = provider.value
        model_name = model.value
        
        try:
            results["summary"]["models_tested"].add(f"{provider_name}:{model_name}")
            
            # Test each formatting type with one language pair
            for format_type, test_data in TEST_CASES.items():
                # Use the first language pair
                source_lang, target_lang = LANGUAGE_PAIRS[0]
                
                test_result = {
                    "provider": provider_name,
                    "model": model_name,
                    "format_type": format_type,
                    "source_lang": source_lang.value,
                    "target_lang": target_lang.value,
                    "domain": test_data["domain"],
                    "text_length": len(test_data["text"]),
                    "passed": False,
                    "details": None,
                    "error": None
                }
                
                logger.info(f"Testing {provider_name}:{model_name} with {format_type} format")
                
                try:
                    # Perform the translation
                    result = await translator.translate(
                        text=test_data["text"],
                        source_lang=source_lang.value,
                        target_lang=target_lang.value,
                        domain=test_data["domain"],
                        quality_level="high",
                        preserve_formatting=True,
                        provider=provider.value,
                        model=model.value
                    )
                    
                    # Check format preservation
                    passed, checks = check_format_preservation(test_data["text"], result["translated_text"])
                    
                    # Update test results
                    test_result["passed"] = passed
                    test_result["details"] = checks
                    
                    if passed:
                        results["summary"]["passed"] += 1
                    else:
                        results["summary"]["failed"] += 1
                        
                    results["summary"]["total"] += 1
                    
                except Exception as e:
                    test_result["error"] = str(e)
                    results["summary"]["failed"] += 1
                    results["summary"]["total"] += 1
                    logger.error(f"Error testing {provider_name}:{model_name}: {str(e)}")
                
                results["tests"].append(test_result)
                
                # Delay between tests to avoid rate limiting
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in model initialization for {provider_name}:{model_name}: {str(e)}")
    
    # Convert set to list for JSON serialization
    results["summary"]["models_tested"] = list(results["summary"]["models_tested"])
    
    # Save results to file
    with open('format_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info(f"Tests completed: {results['summary']['total']} tests run")
    logger.info(f"Passed: {results['summary']['passed']}, Failed: {results['summary']['failed']}")
    
    # Assert overall success rate
    success_rate = results["summary"]["passed"] / results["summary"]["total"] if results["summary"]["total"] > 0 else 0
    assert success_rate >= 0.7, f"Format preservation tests failed with success rate of {success_rate:.2f}"

# Additional test to focus on complex mixed formatting with all language pairs for a specific model
@pytest.mark.asyncio
async def test_mixed_formatting_languages():
    """Test the complex mixed formatting across all language pairs for a default model"""
    
    default_provider = Provider.OPENAI
    default_model = Model.GPT_4O
    
    results = []
    
    # Use the mixed formatting test case
    test_data = TEST_CASES["mixed_formatting"]
    
    for source_lang, target_lang in LANGUAGE_PAIRS:
        test_result = {
            "provider": default_provider.value,
            "model": default_model.value,
            "format_type": "mixed_formatting",
            "source_lang": source_lang.value,
            "target_lang": target_lang.value,
            "passed": False,
            "details": None,
            "error": None
        }
        
        logger.info(f"Testing language pair {source_lang.value}->{target_lang.value} with mixed formatting")
        
        try:
            # Perform the translation
            result = await translator.translate(
                text=test_data["text"],
                source_lang=source_lang.value,
                target_lang=target_lang.value,
                domain=test_data["domain"],
                quality_level="high",
                preserve_formatting=True,
                provider=default_provider.value,
                model=default_model.value
            )
            
            # Check format preservation
            passed, checks = check_format_preservation(test_data["text"], result["translated_text"])
            
            # Update test results
            test_result["passed"] = passed
            test_result["details"] = checks
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"Error testing language pair {source_lang.value}->{target_lang.value}: {str(e)}")
        
        results.append(test_result)
        
        # Delay between tests
        await asyncio.sleep(1)
    
    # Save results to file
    with open('language_pair_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Assert that all language pairs maintain formatting
    passed_count = sum(1 for result in results if result["passed"])
    assert passed_count == len(LANGUAGE_PAIRS), f"Only {passed_count}/{len(LANGUAGE_PAIRS)} language pairs maintained formatting"

# Test for domain-specific formatting differences
@pytest.mark.asyncio
async def test_domain_formatting():
    """Test how domain-specific formatting is preserved across domains"""
    
    # Use Claude for this test as it's good with domain contexts
    provider = Provider.ANTHROPIC
    model = Model.CLAUDE_3_SONNET
    
    results = []
    
    # Use the bullet_lists format for healthcare and numbered_lists for insurance
    healthcare_test = TEST_CASES["bullet_lists"]
    insurance_test = TEST_CASES["numbered_lists"]
    
    # Use English to Spanish for both
    source_lang, target_lang = LanguageCode.ENGLISH, LanguageCode.SPANISH
    
    # Test healthcare domain
    healthcare_result = {
        "domain": "healthcare",
        "passed": False,
        "details": None,
        "error": None
    }
    
    logger.info(f"Testing healthcare domain formatting")
    
    try:
        # Perform the translation
        result = await translator.translate(
            text=healthcare_test["text"],
            source_lang=source_lang.value,
            target_lang=target_lang.value,
            domain="healthcare",
            quality_level="high",
            preserve_formatting=True,
            provider=provider.value,
            model=model.value
        )
        
        # Check format preservation
        passed, checks = check_format_preservation(healthcare_test["text"], result["translated_text"])
        
        # Update test results
        healthcare_result["passed"] = passed
        healthcare_result["details"] = checks
        
    except Exception as e:
        healthcare_result["error"] = str(e)
        logger.error(f"Error testing healthcare domain: {str(e)}")
    
    results.append(healthcare_result)
    
    # Delay between tests
    await asyncio.sleep(1)
    
    # Test insurance domain
    insurance_result = {
        "domain": "insurance",
        "passed": False,
        "details": None,
        "error": None
    }
    
    logger.info(f"Testing insurance domain formatting")
    
    try:
        # Perform the translation
        result = await translator.translate(
            text=insurance_test["text"],
            source_lang=source_lang.value,
            target_lang=target_lang.value,
            domain="insurance",
            quality_level="high",
            preserve_formatting=True,
            provider=provider.value,
            model=model.value
        )
        
        # Check format preservation
        passed, checks = check_format_preservation(insurance_test["text"], result["translated_text"])
        
        # Update test results
        insurance_result["passed"] = passed
        insurance_result["details"] = checks
        
    except Exception as e:
        insurance_result["error"] = str(e)
        logger.error(f"Error testing insurance domain: {str(e)}")
    
    results.append(insurance_result)
    
    # Save results to file
    with open('domain_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Assert that both domains maintain formatting
    assert all(result["passed"] for result in results), "Domain-specific formatting tests failed"

# Test for comparing format preservation with preserve_formatting=True vs False
@pytest.mark.asyncio
async def test_preserve_formatting_flag():
    """Test the impact of preserve_formatting flag on format preservation"""
    
    provider = Provider.OPENAI
    model = Model.GPT_4O
    
    results = []
    
    # Use the tables format
    test_data = TEST_CASES["tables"]
    
    # Use English to Spanish
    source_lang, target_lang = LanguageCode.ENGLISH, LanguageCode.SPANISH
    
    for preserve_flag in [True, False]:
        test_result = {
            "preserve_formatting": preserve_flag,
            "passed": False,
            "details": None,
            "error": None
        }
        
        logger.info(f"Testing preserve_formatting={preserve_flag}")
        
        try:
            # Perform the translation
            result = await translator.translate(
                text=test_data["text"],
                source_lang=source_lang.value,
                target_lang=target_lang.value,
                domain=test_data["domain"],
                quality_level="high",
                preserve_formatting=preserve_flag,
                provider=provider.value,
                model=model.value
            )
            
            # Check format preservation
            passed, checks = check_format_preservation(test_data["text"], result["translated_text"])
            
            # Update test results
            test_result["passed"] = passed
            test_result["details"] = checks
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"Error testing preserve_formatting={preserve_flag}: {str(e)}")
        
        results.append(test_result)
        
        # Delay between tests
        await asyncio.sleep(1)
    
    # Save results to file
    with open('preserve_formatting_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # The true flag should pass, while we don't have expectations for the false flag
    assert results[0]["passed"], "Format preservation with preserve_formatting=True failed"

# Script to run directly without pytest
if __name__ == "__main__":
    # Set up asyncio event loop
    loop = asyncio.get_event_loop()
    
    # Run the tests
    print("Running format preservation tests...")
    loop.run_until_complete(test_all_formats_all_models())
    print("Running language pair tests...")
    loop.run_until_complete(test_mixed_formatting_languages())
    print("Running domain formatting tests...")
    loop.run_until_complete(test_domain_formatting())
    print("Running preserve_formatting flag tests...")
    loop.run_until_complete(test_preserve_formatting_flag())
    
    print("All tests completed! Results saved to JSON files.") 