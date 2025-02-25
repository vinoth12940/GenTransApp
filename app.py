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

        # Define model context windows and output token limits
        self.model_limits = {
            "anthropic": {
                "claude-3-5-sonnet-20241022": {"input_tokens": 200000, "max_output_tokens": 4096},
                "claude-3-5-haiku-20241022": {"input_tokens": 200000, "max_output_tokens": 4096},
                # Default for any other Claude model
                "default": {"input_tokens": 100000, "max_output_tokens": 4096}
            },
            "openai": {
                "gpt-4o": {"input_tokens": 128000, "max_output_tokens": 16384},
                "gpt-4o-mini": {"input_tokens": 128000, "max_output_tokens": 16384},
                # Default for any other OpenAI model
                "default": {"input_tokens": 16000, "max_output_tokens": 4096}
            },
            "google": {
                "gemini-2.0-flash": {"input_tokens": 128000, "max_output_tokens": 8192},
                "gemini-1.5-pro": {"input_tokens": 1000000, "max_output_tokens": 8192},
                "gemini-1.5-flash": {"input_tokens": 128000, "max_output_tokens": 8192},
                # Default for any other Google model
                "default": {"input_tokens": 32000, "max_output_tokens": 4096}
            },
            "groq": {
                "llama-3.3-70b-versatile": {"input_tokens": 8192, "max_output_tokens": 8192},
                "llama-3.3-70b-specdec": {"input_tokens": 8192, "max_output_tokens": 8192},
                # Default for any other Groq model
                "default": {"input_tokens": 8192, "max_output_tokens": 4096}
            },
            "deepseek": {
                "deepseek-chat": {"input_tokens": 32000, "max_output_tokens": 8192},
                # Default for any other DeepSeek model
                "default": {"input_tokens": 8192, "max_output_tokens": 4096}
            },
            "cohere": {
                "command-r-plus-08-2024": {"input_tokens": 128000, "max_output_tokens": 4096},
                # Default for any other Cohere model
                "default": {"input_tokens": 32000, "max_output_tokens": 2048}
            },
            "huggingface": {
                "mistralai/Mistral-7B-Instruct-v0.3": {"input_tokens": 32000, "max_output_tokens": 8192},
                # Default for any other Hugging Face model
                "default": {"input_tokens": 4096, "max_output_tokens": 4096}
            }
        }
        
    def _get_model_limits(self, provider: str, model: str) -> Dict:
        """Get input and output token limits for a specific model"""
        provider_limits = self.model_limits.get(provider, {})
        model_specific_limits = provider_limits.get(model, provider_limits.get("default", {
            "input_tokens": 4096,
            "max_output_tokens": 2048
        }))
        return model_specific_limits
        
    def _get_model_config(self, provider: str, model: str, quality_level: str) -> Dict:
        """Get model-specific configuration"""
        # Base configuration
        config = {
            "temperature": 0.3 if quality_level == "high" else 0.5,
            "max_tokens": 2000,
            "stream": False
        }
        
        # Get model-specific limits
        model_limits = self._get_model_limits(provider, model)
        
        # Provider-specific adjustments
        if provider == "openai":
            # Use model-specific limit, fall back to a safe default
            config["max_tokens"] = model_limits["max_output_tokens"]
            config["top_p"] = 1.0
            
        elif provider == "anthropic":
            # Use model-specific limit, fall back to a safe default
            config["max_tokens"] = min(model_limits["max_output_tokens"], 4096)
            config["top_p"] = 1.0
                
        elif provider == "google":
            config["max_output_tokens"] = config.pop("max_tokens")  # Gemini uses different parameter name
            config["max_output_tokens"] = model_limits["max_output_tokens"]
            config["top_p"] = 1.0
                
        elif provider == "groq":
            config["max_tokens"] = model_limits["max_output_tokens"]
            config["temperature"] = max(config["temperature"], 1e-8)  # Groq requires temperature > 0
            config["top_p"] = 1.0

        # Add DeepSeek configuration (OpenAI-compatible)
        elif provider == "deepseek":
            config["max_tokens"] = model_limits["max_output_tokens"]
            config["temperature"] = max(config["temperature"], 0.01)
            config["top_p"] = 1.0
        
        # Update Hugging Face configuration
        elif provider == "huggingface":
            return {
                "max_tokens": model_limits["max_output_tokens"],
                "stop": ["###", "Notes:", "Important:"],  # Add stopping criteria
                "temperature": 0.1  # Lower temperature for strict compliance
            }
            
        # Add Cohere configuration
        elif provider == "cohere":
            config["max_tokens"] = model_limits["max_output_tokens"]
            if "command-r-plus" in model:
                config["temperature"] = 0.3  # Lower temp for factual responses
            # Remove unsupported parameters
            config.pop("stream", None)
            config.pop("top_p", None)
        
        return config
        
    def _estimate_token_count(self, text: str) -> int:
        """Roughly estimate token count for a text"""
        # Simple estimation: ~4 characters per token for English
        # This is a rough estimate and may need adjustment for other languages
        return len(text) // 4

    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 8000) -> List[str]:
        """Split text into manageable chunks for processing"""
        # If text is small enough, return as is
        if self._estimate_token_count(text) <= max_chunk_size:
            return [text]
            
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = self._estimate_token_count(paragraph)
            
            # If a single paragraph is too big, split it by sentence
            if paragraph_size > max_chunk_size:
                sentences = paragraph.replace('. ', '.\n').split('\n')
                for sentence in sentences:
                    sentence_size = self._estimate_token_count(sentence)
                    
                    if current_size + sentence_size <= max_chunk_size:
                        current_chunk += sentence + ' '
                        current_size += sentence_size
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ' '
                        current_size = sentence_size
            
            # Normal case: add paragraphs until we reach chunk size
            elif current_size + paragraph_size <= max_chunk_size:
                current_chunk += paragraph + '\n\n'
                current_size += paragraph_size
            else:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
                current_size = paragraph_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
        
    def _merge_translations(self, translations: List[str]) -> str:
        """Merge multiple translated chunks back into a single text"""
        # Clean each chunk of potential continuation markers
        cleaned_translations = []
        
        continuation_markers = [
            "[Continúa la traducción", 
            "[Continua la traducción",
            "¿Desea que continúe",
            "Would you like me to continue",
            "[Continuación",
            "[Continuation",
            "Nota:",
            "Note:",
            "[Continued",
            "[To be continued"
        ]
        
        for chunk in translations:
            # Clean each chunk
            cleaned_chunk = chunk.strip()
            
            # Remove any continuation markers at the end of chunks
            for marker in continuation_markers:
                if marker.lower() in cleaned_chunk.lower():
                    # Find the position of the marker
                    pos = cleaned_chunk.lower().find(marker.lower())
                    # Remove everything from the marker to the end
                    cleaned_chunk = cleaned_chunk[:pos].strip()
            
            cleaned_translations.append(cleaned_chunk)
        
        # Join chunks with double newlines for proper separation
        return '\n\n'.join(cleaned_translations)
        
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
        
        # Use provided provider/model or defaults
        provider = provider.value if isinstance(provider, Provider) else provider or self.default_provider
        model_name = model.value if isinstance(model, Model) else model or self.default_model
        
        # Get model limits
        model_limits = self._get_model_limits(provider, model_name)
        
        # Check if document is too large for context window and needs chunking
        estimated_tokens = self._estimate_token_count(text)
        max_input_size = model_limits["input_tokens"] // 2  # Use half of context for input to be safe
        
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
   - Questions to the user like "Would you like me to continue?"
   - Statements about length or continuation
   - ANY text that isn't part of the actual translation
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
3. NEVER add:
   - Introductory phrases like "Here is the translation:"
   - Disclaimer text
   - Quality assessments
   - Text like "[Continued due to length...]"
   - Questions like "Would you like me to continue with the rest of the translation?"
   - ANY indication that the translation is incomplete
4. If facing untranslatable content:
   - Keep original term with [bracketed] explanation in target language
5. Response must be EXACTLY equivalent in meaning to source text
6. When translating lists or tables:
   - Maintain exact same number of items/rows
   - Preserve bullet type (•, *, -, etc.)
   - Keep same indentation levels
   - Maintain table columns and alignment
7. IMPORTANT: You MUST translate the ENTIRE text provided. Do not stop early or indicate continuation.
8. Remember: This is an API call, not a conversation. The complete translation is required in a single response.
"""
        else:
            formatting_instructions = ""

        # Handle large documents with chunking if needed
        if estimated_tokens > max_input_size:
            logger.info(f"Large document detected ({estimated_tokens} estimated tokens). Using chunking.")
            
            # Split text into manageable chunks
            chunks = self._split_text_into_chunks(text, max_chunk_size=max_input_size)
            logger.info(f"Split document into {len(chunks)} chunks for processing")
            
            all_translations = []
            chunk_system_prompt = f"""{DOMAIN_CONTEXTS[domain]}

{formatting_instructions}
Input: {LANGUAGE_NAMES[source_lang]}
Output: {LANGUAGE_NAMES[target_lang]}
Quality Level: {quality_level.upper()}

IMPORTANT INSTRUCTIONS FOR CHUNK TRANSLATION:
1. This is chunk {{chunk_num}} of {{total_chunks}} from a larger document.
2. Translate ONLY the text provided in this chunk completely. 
3. Do not reference other chunks or indicate that this is a partial translation.
4. Do not include ANY meta-commentary about:
   - The chunk being part of a larger document
   - Whether the translation is complete or needs continuation
   - If you're running out of space or tokens
5. ABSOLUTELY NEVER add phrases like:
   - "[Continúa la traducción...]" 
   - "Would you like me to continue?"
   - "[Continued in next chunk...]"
   - Any similar phrases indicating continuation or incompleteness
6. Your output should contain ONLY the translated text - nothing else.
7. This is an API call, not a conversation. The user will not see your response directly.
8. Each chunk will be automatically merged with others - do not add text that would interfere with merging.

THIS IS CRITICAL: NEVER include any continuation or meta-text that would break the flow between chunks."""

            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_prompt = chunk_system_prompt.format(
                    chunk_num=i+1, 
                    total_chunks=len(chunks)
                )
                
                user_message = f"""SOURCE TEXT TO TRANSLATE (CHUNK {i+1}/{len(chunks)}):
{chunk}"""

                try:
                    # Get configuration for this model
                    config = self._get_model_config(provider, model_name, quality_level)
                    
                    # Format model name for AISuite
                    full_model_name = f"{provider}:{model_name}"
                    logger.info(f"Translating chunk {i+1}/{len(chunks)} with {full_model_name}")
                    
                    # Make API call through AISuite
                    if provider == "huggingface":
                        # For HF models, combine system prompt with user message
                        messages = [{
                            "role": "user",
                            "content": f"{chunk_prompt}\n\n{user_message}\n\nTRANSLATED OUTPUT:"
                        }]
                    else:
                        messages = [
                            {"role": "system", "content": chunk_prompt},
                            {"role": "user", "content": user_message}
                        ]
                    
                    # Perform translation for this chunk
                    response = self.client.chat.completions.create(
                        model=full_model_name,
                        messages=messages,
                        **config
                    )
                    
                    chunk_translation = response.choices[0].message.content.strip()
                    all_translations.append(chunk_translation)
                    
                except Exception as e:
                    logger.error(f"Error translating chunk {i+1}: {str(e)}")
                    # Add placeholder for failed chunk to maintain order
                    all_translations.append(f"[Translation error in chunk {i+1}]")
            
            # Merge all translations
            translated_text = self._merge_translations(all_translations)
            
        else:
            # Standard translation for documents that fit in context window
            system_prompt = f"""{DOMAIN_CONTEXTS[domain]}

{formatting_instructions}
Input: {LANGUAGE_NAMES[source_lang]}
Output: {LANGUAGE_NAMES[target_lang]}
Quality Level: {quality_level.upper()}

IMPORTANT INSTRUCTIONS FOR DOCUMENT TRANSLATION:
1. Translate the COMPLETE document from beginning to end.
2. Do not include ANY meta-commentary about:
   - Whether the translation is complete or needs continuation
   - If you're running out of space or tokens
3. ABSOLUTELY NEVER add phrases like:
   - "[Continúa la traducción...]"
   - "Would you like me to continue?"
   - "[Continued in next section...]"
   - Any similar phrases indicating continuation or incompleteness
4. Your output should contain ONLY the translated text - nothing else.
5. This is an API call, not a conversation. The user will not see your response directly.

THIS IS CRITICAL: NEVER include any continuation or meta-text in your translation."""

            user_message = f"""SOURCE TEXT TO TRANSLATE:
{text}"""

            try:
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
            
            except Exception as e:
                logger.error(f"Translation error with {provider}:{model_name} - {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Translation failed: {str(e)}"
                )
            
        # Validate translation if quality_level is "high"
        validation_result = None
        if quality_level == "high":
            # Get text sample for validation (may be too large for validator)
            validation_text = text
            validation_translated = translated_text
            
            # If text is very large, just validate a representative sample
            max_validation_size = 8000  # tokens
            if self._estimate_token_count(text) > max_validation_size:
                # Take first and last portions to validate
                validation_chunks = self._split_text_into_chunks(text, max_chunk_size=max_validation_size//2)
                if len(validation_chunks) > 1:
                    validation_text = validation_chunks[0] + "\n\n[...]\n\n" + validation_chunks[-1]
                    
                # Match with corresponding parts of translation
                trans_chunks = self._split_text_into_chunks(translated_text, max_chunk_size=max_validation_size//2)
                if len(trans_chunks) > 1:
                    validation_translated = trans_chunks[0] + "\n\n[...]\n\n" + trans_chunks[-1]
            
            best_score = 0
            best_translation = translated_text
            validation_history = []
            remaining_attempts = 3  # Total attempts including initial
            improvement_threshold = 98.0

            for attempt in range(remaining_attempts):
                # Validate current translation
                score, feedback, issues = await self.validator.validate_translation(
                    validation_text,
                    validation_translated,
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
                validation_text,
                validation_translated,  # Ensure we validate the BEST version
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