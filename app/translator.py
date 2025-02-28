"""
Unified Translator module for multi-model, multi-provider translation.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import aisuite as ai

from app.constants import LANGUAGE_NAMES, DOMAIN_CONTEXTS
from app.validator import TranslationValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            
        # Special handling for Google Vertex AI setup
        self._setup_google_vertex_ai()
        
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
        
    def _setup_google_vertex_ai(self):
        """
        Set up Google Vertex AI integration based on aisuite guide:
        https://github.com/andrewyng/aisuite/blob/main/guides/google.md
        """
        # Check if the required environment variables for Google Vertex AI are set
        google_project_id = os.getenv("GOOGLE_PROJECT_ID")
        google_region = os.getenv("GOOGLE_REGION")
        google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Log whether Google Vertex AI is properly configured
        if all([google_project_id, google_region, google_credentials]):
            logger.info(f"Google Vertex AI configured with project ID: {google_project_id}, region: {google_region}")
            
            # Ensure the credentials file exists
            if not os.path.exists(google_credentials):
                logger.warning(f"Google credentials file not found at: {google_credentials}")
            else:
                logger.info("Google credentials file found")
                
            # Set these environment variables for aisuite/vertexai
            os.environ["GOOGLE_PROJECT_ID"] = google_project_id
            os.environ["GOOGLE_REGION"] = google_region
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials
            
            try:
                # Optional: Try importing vertexai to check if it's installed
                import importlib
                vertexai_spec = importlib.util.find_spec("vertexai")
                if vertexai_spec is None:
                    logger.warning("vertexai package not found. Please install with: pip install vertexai")
                else:
                    logger.info("vertexai package found")
            except ImportError:
                logger.warning("Could not check for vertexai package. Please ensure it's installed.")
        else:
            missing_vars = []
            if not google_project_id: missing_vars.append("GOOGLE_PROJECT_ID")
            if not google_region: missing_vars.append("GOOGLE_REGION")
            if not google_credentials: missing_vars.append("GOOGLE_APPLICATION_CREDENTIALS")
            
            logger.warning(f"Google Vertex AI not fully configured. Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("Google models may not work properly. Please refer to: https://github.com/andrewyng/aisuite/blob/main/guides/google.md")
        
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
        provider = provider.value if hasattr(provider, 'value') else provider or self.default_provider
        model_name = model.value if hasattr(model, 'value') else model or self.default_model
        
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
                raise Exception(f"Translation failed: {str(e)}")
            
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