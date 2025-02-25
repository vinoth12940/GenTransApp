from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Literal, Tuple
import os
from dotenv import load_dotenv
from enum import Enum
import json
import uuid
from datetime import datetime
import logging
import asyncio
import aisuite as ai

# Create FastAPI app instance
app = FastAPI(
    title="Insurance & Healthcare Translation API",
    description="""
    A specialized translation service for P&C Insurance and Healthcare domains.
    
    ## Features
    * Support for multiple language models (OpenAI, Anthropic, Google, Groq)
    * Domain-specific translations (Healthcare and Insurance)
    * Multiple language pairs (English, Spanish, Chinese)
    * Quality control levels (high, medium, draft)
    
    ## Available Models
    
    ### OpenAI Models
    * GPT-4o (128k context, 16k output)
    * GPT-4o Mini (128k context, 16k output)
    
    ### Anthropic Models
    * Claude 3.5 Sonnet (Latest version)
    * Claude 3.5 Haiku (Fast and efficient)
    
    ### Google Models
    * Gemini 2.0 Flash (Latest and fastest)
    * Gemini 1.5 Pro (Most capable)
    * Gemini 1.5 Flash (Balanced)
    
    ### Groq Models
    * Llama 3.3 70B Versatile (8k context, 4k output)
    * Llama 3.3 70B SpecDec (8k context, 4k output)
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {
            "name": "Translation",
            "description": "Translation operations with support for multiple models and domains",
        },
        {
            "name": "System",
            "description": "System information and health check endpoints",
        },
    ]
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LanguageCode(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    CHINESE = "zh"

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    COHERE = "cohere"
    GOOGLE = "google"
    MISTRAL = "mistral"
    AWS = "aws"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    DEEPSEEK = "deepseek"  # Added DeepSeek provider

class Model(str, Enum):
    # Anthropic Models
    CLAUDE_3_SONNET = "claude-3-5-sonnet-20241022"  # Latest version
    CLAUDE_3_HAIKU = "claude-3-5-haiku-20241022"  # Fast and efficient
    
    # OpenAI Models
    GPT_4O = "gpt-4o"  # 128k context, 16k output
    GPT_4O_MINI = "gpt-4o-mini"  # 128k context, 16k output
    
    # Google Models
    GEMINI_2_FLASH = "gemini-2.0-flash"  # Latest and fastest
    GEMINI_1_5_PRO = "gemini-1.5-pro"  # Most capable
    GEMINI_1_5_FLASH = "gemini-1.5-flash"  # Balanced
    
    # Groq Models
    LLAMA_3_70B = "llama-3.3-70b-versatile"  # 8k context, 4k output
    LLAMA_3_8B = "llama-3.3-70b-specdec"     # 8k context, 4k output

    # Cohere Models
    COHERE_COMMAND_R_PLUS = "command-r-plus-08-2024"  # From cohere.md
    
    # DeepSeek Models
    DEEPSEEK_CHAT = "deepseek-chat"  # From deepseek.md
    
    # Hugging Face Models
    HUGGINGFACE_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"

class TranslationRequest(BaseModel):
    text: str
    source_language: LanguageCode
    target_language: LanguageCode
    domain: str = "healthcare"  # or "insurance"
    preserve_formatting: bool = True
    quality_level: str = "high"  # high, medium, draft
    provider: Optional[Provider] = None
    model: Optional[Model] = None

class ValidationResult(BaseModel):
    quality_score: float
    feedback: str
    issues: List[str]

class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    domain: str
    request_id: str
    provider: str
    model: str
    latency_ms: float
    validation: Optional[ValidationResult] = None

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "zh": "Chinese (Simplified)"
}

DOMAIN_CONTEXTS = {
    "healthcare": """You are a certified medical translation specialist with expertise in:
    - Medical terminology (ICD-10, CPT codes, clinical terms)
    - Patient records and clinical documentation
    - Pharmaceutical documentation and drug information
    - Medical device specifications
    - Clinical trial protocols
    - HIPAA-compliant language handling
    
    Requirements:
    1. Maintain exact meaning of medical terminology - never approximate
    2. Preserve numerical values (dosages, measurements, lab values)
    3. Keep original formatting of medical forms and tables
    4. Use official terminology from:
       - WHO International Classifications
       - US National Library of Medicine MeSH
       - Target country's medical regulatory bodies
    5. Handle abbreviations carefully (expand with original in parentheses first occurrence)
    6. Maintain patient confidentiality markers
    7. Cross-cultural adaptation for medical concepts""",
    
    "insurance": """You are a licensed insurance translation expert with expertise in:
    - Policy wording and coverage terms
    - Claims documentation and adjuster reports
    - Underwriting guidelines
    - Regulatory compliance documents
    - Insurance contracts and endorsements
    - P&C insurance specific terminology:
      * Auto insurance (collision, comprehensive, liability)
      * Property insurance (HO-3, HO-6 policies)
      * Commercial general liability
      * Reinsurance treaties
    
    Requirements:
    1. Maintain legal precision of policy language
    2. Preserve exact meanings of coverage terms and exclusions
    3. Keep original formatting of tables, definitions sections, and schedules
    4. Use official terminology from:
       - NAIC model laws
       - ISO standard forms
       - Target jurisdiction's insurance regulations
    5. Handle numbers/dates/currency with absolute accuracy
    6. Maintain document hierarchy (WHEREAS clauses, numbered paragraphs)
    7. Cross-jurisdictional compliance checks"""
}

class TranslationValidator:
    def __init__(self):
        self.client = ai.Client()
        self.model = "anthropic:claude-3-5-sonnet-20241022"
        
    async def validate_translation(
        self,
        source_text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
        domain: str
    ) -> Tuple[float, str, List[str]]:
        system_prompt = f"""You are an expert {domain} translation validator with native fluency in both {LANGUAGE_NAMES[source_lang]} and {LANGUAGE_NAMES[target_lang]}.
        
        Evaluation criteria for {domain} translation:
        1. Accuracy (35%):
           - Precise translation of {domain} terminology
           - Preservation of numerical values and measurements
           - Correct handling of abbreviations and codes
           
        2. Fluency (25%):
           - Natural expression in {LANGUAGE_NAMES[target_lang]}
           - Appropriate register for {domain} context
           - Correct grammar and syntax
           
        3. Format Preservation (20%):
           - Exact preservation of bullet points and bullet lists
           - Maintenance of all table structures and alignments
           - Preservation of indentation and paragraph structure
           - Retention of numbered lists and their format
           - Preservation of all special formatting (bold, italic, etc.)
           
        4. Domain Compliance (20%):
           - Adherence to {domain}-specific standards
           - Correct use of regulatory terminology
           - Proper formatting for {domain} documents
        
        Provide a detailed evaluation with:
        1. Quality score (0-100)
        2. Specific feedback on strengths/weaknesses
        3. List of issues that need improvement
        
        Format response as JSON:
        {{
            "score": <number>,
            "feedback": "<detailed_feedback>",
            "issues": [
                "<specific_issue_1>",
                "<specific_issue_2>",
                ...
            ]
        }}"""

        # Add JSON formatting instruction to system prompt
        system_prompt += "\n\nIMPORTANT: Your response MUST be valid JSON with only the specified fields."
        
        user_message = f"""Source {LANGUAGE_NAMES[source_lang]} text:
        {source_text}
        
        Translation in {LANGUAGE_NAMES[target_lang]}:
        {translated_text}
        
        Provide a thorough evaluation following the criteria above."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                # Remove response_format parameter
                max_tokens=4000  # Add token limit for safety
            )
            
            # Add JSON parsing with error handling
            try:
                result = json.loads(response.choices[0].message.content)
                if not all(key in result for key in ["score", "feedback", "issues"]):
                    raise ValueError("Missing required fields in validation response")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse validation JSON: {e.msg}")
                return (0.0, "Validation response format invalid", [])
                
            # Ensure score is within 0-100 range
            score = max(0, min(100, float(result["score"])))
            
            return (
                score,
                result["feedback"],
                result["issues"]
            )
            
        except Exception as e:
            logger.error(f"Translation validation error: {str(e)}")
            return (0.0, f"Validation failed: {str(e)}", [])

class UnifiedTranslator:
    def __init__(self):
        # Initialize default settings
        self.default_provider = os.getenv("DEFAULT_PROVIDER", "openai")
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")  # Default to GPT-4o
        
        # Initialize AISuite client
        self.client = ai.Client()
        
        # Set API keys in environment variables
        for provider, env_var in {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "cohere": "COHERE_API_KEY",
            "huggingface": "HF_TOKEN"
        }.items():
            if api_key := os.getenv(env_var):
                os.environ[env_var] = api_key
                setattr(self.client, f"{provider}_api_key", api_key)

        # Special handling for Cohere API key
        if cohere_key := os.getenv("COHERE_API_KEY"):
            os.environ["CO_API_KEY"] = cohere_key

        # Special handling for HuggingFace tokens
        if hf_token := os.getenv("HF_TOKEN"):
            # Set the token in environment variables as per documentation
            os.environ["HF_TOKEN"] = hf_token
            # Set it directly on the client as per documentation
            setattr(self.client, "huggingface_api_key", hf_token)
        
        # Add validator
        self.validator = TranslationValidator()
        
    def _get_model_config(self, provider: str, model: str, quality_level: str) -> Dict:
        """Get model-specific configuration"""
        # Base configuration
        config = {
            "temperature": 0.3 if quality_level == "high" else 0.5,
            "max_tokens": 2000,
            "stream": False
        }
        
        # Provider-specific adjustments
        if provider == "openai":
            config["max_tokens"] = 16384  # Both gpt-4o and gpt-4o-mini support 16k output
            config["top_p"] = 1.0
            
        elif provider == "anthropic":
            if "sonnet" in model:
                config["max_tokens"] = 4096
            elif "haiku" in model:
                config["max_tokens"] = 2048
            config["top_p"] = 1.0
                
        elif provider == "google":
            config["max_output_tokens"] = config.pop("max_tokens")  # Gemini uses different parameter name
            if "flash" in model:
                config["max_output_tokens"] = 2048  # Conservative limit for flash models
            elif "pro" in model:
                config["max_output_tokens"] = 4096  # Higher limit for pro model
            config["top_p"] = 1.0
                
        elif provider == "groq":
            # Groq models have a max sequence length of 8192
            config["max_tokens"] = min(config["max_tokens"], 4096)  # Set a safe limit
            if "specdec" in model:
                config["max_tokens"] = min(config["max_tokens"], 4096)  # More conservative for specialized models
            config["temperature"] = max(config["temperature"], 1e-8)  # Groq requires temperature > 0
            config["top_p"] = 1.0

        # Add DeepSeek configuration (OpenAI-compatible)
        elif provider == "deepseek":
            config["max_tokens"] = 4096  # Match DeepSeek's recommended limits
            config["temperature"] = max(config["temperature"], 0.01)
            config["top_p"] = 1.0
        
        # Update Hugging Face configuration
        elif provider == "huggingface":
            return {
                "stop": ["###", "Notes:", "Important:"],  # Add stopping criteria
                "temperature": 0.1  # Lower temperature for strict compliance
            }
            
        # Add Cohere configuration
        elif provider == "cohere":
            config["max_tokens"] = 2048
            if "command-r-plus" in model:
                config["temperature"] = 0.3  # Lower temp for factual responses
            # Remove unsupported parameters
            config.pop("stream", None)
            config.pop("top_p", None)
        
        return config
        
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        domain: str,
        quality_level: str,
        preserve_formatting: bool,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict:
        start_time = datetime.now()
        
        # Modified system prompt with strict output formatting instructions
        if preserve_formatting:
            formatting_instructions = """
**CRITICAL OUTPUT FORMATTING RULES:**
1. Return ONLY the translated text without:
   - Additional commentary
   - Explanations of translation choices
   - Formatting instructions
   - Notes about terminology
   - Any meta-text
2. Preserve ALL original formatting including:
   - Line breaks
   - Bullet points (•, *, -, etc.)
   - Bullet lists and nested lists
   - Numbered lists and their indentation
   - Tables and their structure/alignment
   - Indentation and paragraph spacing
   - Headers and subheaders
   - Bold, italic, or other text styles if markers are present
   - Special characters and symbols
   - List item spacing
3. Never add:
   - Introductory phrases like "Here is the translation:"
   - Disclaimer text
   - Quality assessments
4. If facing untranslatable content:
   - Keep original term with [bracketed] explanation in target language
5. Response must be EXACTLY equivalent in meaning to source text
6. When translating lists or tables:
   - Maintain exact same number of items/rows
   - Preserve bullet type (•, *, -, etc.)
   - Keep same indentation levels
   - Maintain table columns and alignment
"""
        else:
            formatting_instructions = ""

        system_prompt = f"""{DOMAIN_CONTEXTS[domain]}

{formatting_instructions}
Input: {LANGUAGE_NAMES[source_lang]}
Output: {LANGUAGE_NAMES[target_lang]}
Quality Level: {quality_level.upper()}"""

        user_message = f"""SOURCE TEXT TO TRANSLATE:
{text}"""

        try:
            # Use provided provider/model or defaults
            provider = provider.value if isinstance(provider, Provider) else provider or self.default_provider
            model_name = model.value if isinstance(model, Model) else model or self.default_model
            
            # Get configuration
            config = self._get_model_config(provider, model_name, quality_level)
            
            # Format model name for AISuite
            full_model_name = f"{provider}:{model_name}"
            logger.info(f"Starting translation with {full_model_name}")
            
            # Make API call through AISuite
            if provider == "huggingface":
                # For HF models, combine system prompt with user message
                messages = [{
                    "role": "user",
                    "content": f"{system_prompt}\n\n{user_message}\n\nTRANSLATED OUTPUT:"
                }]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            
            # Perform initial translation
            response = self.client.chat.completions.create(
                model=full_model_name,
                messages=messages,
                **config
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Validate translation if quality_level is "high"
            validation_result = None
            if quality_level == "high":
                best_score = 0
                best_translation = translated_text
                validation_history = []
                remaining_attempts = 3  # Total attempts including initial
                improvement_threshold = 98.0

                for attempt in range(remaining_attempts):
                    # Validate current translation
                    score, feedback, issues = await self.validator.validate_translation(
                        text,
                        translated_text,
                        source_lang,
                        target_lang,
                        domain
                    )
                    
                    # Track best result
                    if score > best_score:
                        best_score = score
                        best_translation = translated_text
                        validation_result = {
                            "quality_score": score,
                            "feedback": feedback,
                            "issues": issues
                        }
                    
                    # Store validation history for context
                    validation_history.append({
                        "attempt": attempt + 1,
                        "score": score,
                        "issues": issues,
                        "feedback": feedback
                    })

                    # Exit if we meet the threshold
                    if score >= improvement_threshold:
                        break

                    # Prepare improvement prompt with structured feedback
                    improvement_instructions = "\n".join([
                        f"Improvement Required ({issue_num+1}): {issue}"
                        for issue_num, issue in enumerate(issues)
                    ])
                    
                    retry_prompt = f"""TRANSLATION IMPROVEMENT CYCLE - ATTEMPT {attempt+1}
                    
                    Previous Validation Results:
                    - Quality Score: {score}/100
                    - Critical Issues Found: {len(issues)}
                    
                    Required Improvements:
                    {improvement_instructions}
                    
                    Please revise the translation to:
                    1. Address all listed issues specifically
                    2. Maintain exact terminology accuracy
                    3. STRICTLY preserve original formatting:
                       - Ensure all bullet points (•, *, -, etc.) match exactly
                       - Maintain all table structures with same alignment
                       - Keep exactly the same indentation levels
                       - Preserve numbered lists with same numbering style
                       - Maintain paragraph breaks exactly as in source
                    4. Follow these exact steps:
                       a. Review each issue from Required Improvements
                       b. Modify ONLY problematic sections
                       c. Keep properly translated parts unchanged
                       d. Verify against source text format
                       e. Check that ALL formatting elements are preserved
                    
                    Return ONLY the improved translation without commentary or explanation."""

                    messages.extend([
                        {"role": "assistant", "content": translated_text},
                        {"role": "user", "content": retry_prompt}
                    ])
                    
                    # Generate improved translation
                    retry_response = self.client.chat.completions.create(
                        model=full_model_name,
                        messages=messages,
                        **config
                    )
                    translated_text = retry_response.choices[0].message.content.strip()

                # Final validation of best version
                final_score, final_feedback, final_issues = await self.validator.validate_translation(
                    text,
                    best_translation,  # Ensure we validate the BEST version
                    source_lang,
                    target_lang,
                    domain
                )
                
                validation_result = {
                    "quality_score": final_score,  # Use FINAL score here
                    "feedback": f"Iterative improvement process completed ({len(validation_history)} attempts). Best score: {best_score}/100 → Final score: {final_score}/100\n{final_feedback}",
                    "issues": final_issues,
                    "validation_history": [
                        {
                            "attempt": item["attempt"],
                            "score": item["score"],
                            "issues": item["issues"]
                        }
                        for item in validation_history
                    ]
                }
                
                translated_text = best_translation
                
            # Calculate latency and return results
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            
            result = {
                'translated_text': translated_text,
                'provider': provider,
                'model': model_name,
                'latency_ms': latency
            }
            
            if validation_result:
                result['validation'] = validation_result
                
            return result

        except Exception as e:
            logger.error(f"Translation error with {provider}:{model_name} - {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Translation failed: {str(e)}"
            )

translator = UnifiedTranslator()

@app.post("/translate", response_model=TranslationResponse, tags=["Translation"],
    summary="Translate text using specified model",
    description="""
    Translate text between supported languages using the specified model and provider.
    
    **Quality Levels:**
    * high: Most accurate translation with domain-specific terminology
    * medium: Balanced between accuracy and speed
    * draft: Fastest translation, suitable for understanding the general meaning
    
    **Domains:**
    * healthcare: Medical terminology and procedures
    * insurance: Insurance policy and claims terminology
    
    **Format Preservation:**
    When preserve_formatting=true (default), the translation will maintain the exact formatting of the original text:
    * Bullet points and bullet lists (•, *, -, etc.)
    * Nested lists with indentation
    * Numbered lists (1., 2., etc.)
    * Tables with columns and alignment
    * Headers and subheaders
    * Paragraph breaks and spacing
    * Bold, italic, and other text styles
    """,
    response_description="Returns the translated text along with metadata"
)
async def translate_text(
    request: TranslationRequest,
    background_tasks: BackgroundTasks
):
    try:
        # Validate language pair
        if request.source_language == request.target_language:
            raise HTTPException(
                status_code=400,
                detail="Source and target languages must be different"
            )
        
        # Validate domain
        if request.domain not in ["healthcare", "insurance"]:
            raise HTTPException(
                status_code=400,
                detail="Domain must be either 'healthcare' or 'insurance'"
            )

        request_id = str(uuid.uuid4())
        
        # Perform translation
        result = await translator.translate(
            text=request.text,
            source_lang=request.source_language,
            target_lang=request.target_language,
            domain=request.domain,
            quality_level=request.quality_level,
            preserve_formatting=request.preserve_formatting,
            provider=request.provider,
            model=request.model
        )

        # Add async logging task
        background_tasks.add_task(
            logger.info,
            f"Translation completed - RequestID: {request_id}, "
            f"Domain: {request.domain}, Provider: {result['provider']}, Model: {result['model']}"
        )

        # Create ValidationResult instance if validation data exists
        validation = None
        if 'validation' in result:
            validation = ValidationResult(
                quality_score=result['validation']['quality_score'],
                feedback=result['validation']['feedback'],
                issues=result['validation']['issues']
            )

        return TranslationResponse(
            translated_text=result['translated_text'],
            source_language=request.source_language,
            target_language=request.target_language,
            domain=request.domain,
            request_id=request_id,
            provider=result['provider'],
            model=result['model'],
            latency_ms=result['latency_ms'],
            validation=validation  # Include the validation results
        )

    except Exception as e:
        logger.error(f"Translation request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-languages", tags=["System"],
    summary="Get supported languages",
    description="Returns a list of supported languages with their codes and names",
    response_description="List of supported languages"
)
async def get_supported_languages():
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in LANGUAGE_NAMES.items()
        ]
    }

@app.get("/supported-providers", tags=["System"],
    summary="Get supported providers and models",
    description="Returns a list of supported translation providers and their available models",
    response_description="List of supported providers and models"
)
async def get_supported_providers():
    return {
        "providers": [provider.value for provider in Provider],
        "models": [model.value for model in Model]
    }

@app.get("/health", tags=["System"],
    summary="Check API health",
    description="Performs a test translation to verify system health",
    response_description="Health status and test translation metrics"
)
async def health_check():
    try:
        # Test model connectivity with default provider/model
        test_result = await translator.translate(
            text="Hello",
            source_lang="en",
            target_lang="es",
            domain="healthcare",
            quality_level="draft",
            preserve_formatting=True
        )
        
        return {
            "status": "healthy",
            "provider": test_result['provider'],
            "model": test_result['model'],
            "latency_ms": test_result['latency_ms'],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 