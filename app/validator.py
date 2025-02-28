"""
Validation module for translation quality assessment.
"""

import json
import logging
from typing import Tuple, List
import aisuite as ai

from app.constants import LANGUAGE_NAMES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
                max_tokens=4000
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