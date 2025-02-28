"""
Data models for the translator application.
Includes enums, Pydantic models, and other data structures.
"""

from enum import Enum
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel

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
    DEEPSEEK = "deepseek"

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
    COHERE_COMMAND_R_PLUS = "command-r-plus-08-2024"
    
    # DeepSeek Models
    DEEPSEEK_CHAT = "deepseek-chat"
    
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