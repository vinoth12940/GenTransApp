"""
Main FastAPI application for translation service.
"""

import uuid
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from dotenv import load_dotenv

from app.models import TranslationRequest, TranslationResponse, ValidationResult, LanguageCode, Provider, Model
from app.translator import UnifiedTranslator
from app.constants import LANGUAGE_NAMES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize translator
translator = UnifiedTranslator()

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
            validation=validation
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