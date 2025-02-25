# Insurance & Healthcare Translation API

A specialized translation service for P&C Insurance and Healthcare domains using FastAPI and multiple AI providers.

## Features

### Core Capabilities

- Multi-provider AI integration with quality control
- Domain-specific expertise in Healthcare and Insurance
- Support for English, Spanish, and Chinese translations
- Advanced format preservation for complex documents
- Quality validation with detailed feedback
- HIPAA-compliant logging and processing

### Supported AI Providers & Models

#### OpenAI Models

- GPT-4o (128k context, 16k output)
- GPT-4o Mini (128k context, 16k output)

#### Anthropic Models

- Claude 3.5 Sonnet (Latest version)
- Claude 3.5 Haiku (Fast and efficient)

#### Google Models

- Gemini 2.0 Flash (Latest and fastest)
- Gemini 1.5 Pro (Most capable)
- Gemini 1.5 Flash (Balanced)

#### Groq Models

- Llama 3.3 70B Versatile (8k context, 4k output)
- Llama 3.3 70B SpecDec (8k context, 4k output)

#### Additional Providers

- Cohere: Command-R Plus
- DeepSeek: DeepSeek Chat
- HuggingFace: Mistral-7B-Instruct-v0.3

### Domain Expertise

#### Healthcare Translation

- Medical terminology (ICD-10, CPT codes)
- Clinical documentation
- Pharmaceutical information
- Medical device specifications
- Clinical trial protocols
- HIPAA compliance

#### Insurance Translation

- Policy wording and coverage terms
- Claims documentation
- Underwriting guidelines
- Regulatory compliance
- Insurance contracts
- P&C insurance terminology

### Format Preservation

The API includes advanced format preservation capabilities that maintain the exact visual structure of translated documents:

#### Preserved Formatting Elements

- Bullet points and bullet lists (•, *, -, etc.)
- Nested lists with original indentation levels
- Numbered lists and alphanumeric sequences
- Tables with proper alignment and structure
- Headers and subheaders (# and ## markdown)
- Paragraph breaks and spacing
- Bold, italic, and other text styling

#### How It Works

1. Enhanced model instructions enforce strict format preservation
2. Dedicated validation evaluates formatting accuracy (20% of quality score)
3. Format-specific improvement cycles for translations below threshold
4. Cross-model consistency for reliable output across providers

#### Example Use Cases

- Medical forms with complex tables and lists
- Insurance policies with hierarchical formatting
- Regulatory documents requiring exact structural preservation
- Clinical documentation with precise formatting requirements

## System Workflow

### End-to-End System Architecture and Process Flow

```mermaid
flowchart TB
    %% Request/Response
    REQ["1. POST /translate Request"]
    RES["7. API Response"]
  
    %% FastAPI Components
    subgraph API["2. FastAPI Layer"]
        FA["2.1 FastAPI Router"]
        RV["2.2 Request Validator"]
        BT["2.3 Background Tasks"]
    end
  
    %% Translation Components
    subgraph TS["3. Translation Service"]
        PS["3.1 Provider Selection"]
        DC["3.2 Domain Context"]
        TP["3.3 Translation Processing"]
        LOG["3.4 HIPAA Logging"]
    end
  
    %% AI Provider Layer
    subgraph AI["4. AI Providers"]
        OAI["OpenAI GPT-4o"]
        ANT["Claude 3.5"]
        GGL["Gemini"]
        GRQ["Groq Llama"]
        DS["DeepSeek"]
        CH["Cohere"]
        HF["Mistral-7B"]
    end

    %% Quality Control
    subgraph QC["5. Quality Control"]
        VAL["5.1 Validation"]
        subgraph SC["5.2 Quality Scoring"]
            AC1["Accuracy (40%)"]
            FA1["Fluency (30%)"]
            DM["Domain Match (30%)"]
        end
        RTY["5.3 Retry Logic"]
    end
  
    %% Response Preparation
    subgraph RP["6. Response Preparation"]
        FMT["6.1 Format Response"]
        UUID["6.2 Generate UUID"]
        LAT["6.3 Calculate Latency"]
    end

    %% Main Flow with Steps
    REQ --> |"1. Submit Request"| FA
    FA --> |"2. Validate"| RV
    RV --> |"3. Select Provider"| PS
    PS --> |"3.2 Load Context"| DC
    DC --> |"3.3 Process"| TP
  
    %% AI Integration
    TP --> |"4. Translate"| AI
    AI --> |"4.2 Return Translation"| TP
  
    %% Quality Flow
    TP --> |"5. Validate"| VAL
    VAL --> |"5.2 Score"| SC
    AC1 & FA1 & DM --> RTY
    RTY --> |"Score < 85%"| TP
    RTY --> |"Score ≥ 85%"| FMT
  
    %% Response Generation
    TP --> |"3.4 Log"| LOG
    LOG --> |"Background"| BT
    VAL --> |"6. Prepare"| FMT
    FMT --> |"6.2"| UUID
    UUID --> |"6.3"| LAT
    LAT --> |"7. Return"| RES

    %% Node styling
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px
    classDef endpoint fill:#e8f4f8,stroke:#156b87
    classDef api fill:#fdebd0,stroke:#ca6f1e
    classDef service fill:#d4efdf,stroke:#196f3d
    classDef ai fill:#f5b7b1,stroke:#943126
    classDef quality fill:#d1f2eb,stroke:#148f77
    classDef logging fill:#e8daef,stroke:#6c3483
    classDef response fill:#fff7e6,stroke:#d4b106

    %% Apply styles
    class REQ,RES endpoint
    class FA,RV,BT api
    class PS,DC,TP service
    class OAI,ANT,GGL,GRQ,DS,CH,HF ai
    class VAL,AC1,FA1,DM,RTY quality
    class LOG logging
    class FMT,UUID,LAT response
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
# Create .env file with your API keys
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
GROQ_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
COHERE_API_KEY=your_key
HF_TOKEN=your_key

# Optional configuration
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
```

4. Run the application:

```bash
uvicorn app:app --reload
```

## API Documentation

### Interactive Swagger UI Documentation

The API comes with built-in interactive documentation powered by Swagger UI, available at:

```
http://localhost:8000/docs
```

Features of the Swagger UI:

- Interactive API testing interface
- Detailed schema information
- Request/response examples
- Authentication setup
- Model definitions
- Try-it-out functionality

Alternative API documentation (ReDoc) is available at:

```
http://localhost:8000/redoc
```

### OpenAPI Specification

The complete OpenAPI specification is available at:

```
http://localhost:8000/openapi.json
```

### Using Swagger UI for Testing

1. Navigate to `http://localhost:8000/docs` in your web browser
2. The Swagger UI interface will show all available endpoints
3. Click on an endpoint to expand it
4. Click the "Try it out" button
5. Fill in the required parameters:
   - For POST requests, you can use the provided JSON schema
   - For GET requests, fill in any query parameters
6. Click "Execute" to send the request
7. View the:
   - Request URL
   - Response status code
   - Response body
   - Response headers

Example using the translation endpoint:

1. Expand the POST /translate endpoint
2. Click "Try it out"
3. Use this sample request:

```json
{
    "text": "Patient shows signs of acute myocardial infarction",
    "source_language": "en",
    "target_language": "es",
    "domain": "healthcare",
    "quality_level": "high",
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022"
}
```

4. Click "Execute"

## Format Preservation Examples

### Using the Postman Collection

The project includes a comprehensive Postman collection (`postman_collection.json`) with examples demonstrating format preservation capabilities:

1. **Healthcare Format Test**

   - Preserves bullet points, nested lists, tables and markdown formatting
   - Maintains complex medical document structure
   - Demonstrates Claude 3.5 Sonnet's format handling capabilities
2. **Insurance Document Format Test**

   - Preserves policy structure with headers, bullets, and tables
   - Maintains exact monetary values and coverage details
   - Shows GPT-4o's capabilities for complex insurance documents

### Example Request (Complex Formatting)

```json
{
  "text": "# Patient Health Summary\n\n## Medical History\n* **Hypertension**: Diagnosed in 2018\n* **Type 2 Diabetes**: Diagnosed in 2019\n  * HbA1c: 7.2% (last checked 2023-05-15)\n  * Currently managed with Metformin 1000mg twice daily\n* **Allergies**:\n  * Penicillin - Severe rash\n  * Shellfish - Anaphylaxis\n\n## Current Medications\n| Medication | Dosage | Frequency | Purpose |\n|------------|--------|-----------|--------|\n| Lisinopril | 10mg | Once daily | Hypertension |\n| Metformin | 1000mg | Twice daily | Diabetes |\n| Aspirin | 81mg | Once daily | Cardiac prophylaxis |\n\n## Recommended Follow-up\n1. Schedule endocrinology appointment within 3 months\n2. Blood pressure monitoring weekly\n3. Complete lipid panel at next visit",
  "source_language": "en",
  "target_language": "es",
  "domain": "healthcare",
  "quality_level": "high",
  "preserve_formatting": true,
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022"
}
```

### Tips for Format Preservation

1. Always set `preserve_formatting: true` to maintain document structure
2. Use `quality_level: "high"` for translations requiring precise formatting
3. Consider provider selection based on format complexity:
   - Claude models excel at complex markdown
   - GPT-4o handles tables and lists exceptionally well
   - Gemini models maintain spacing and indentation effectively
4. Run format preservation tests to determine the best model for your specific formatting needs

## Testing Framework

The API includes a comprehensive testing framework for validating format preservation across different models and scenarios:

### Format Preservation Tests

- Tests all providers and models with various formatting elements
- Validates format preservation across all language pairs
- Compares performance across different domains
- Verifies the impact of the `preserve_formatting` flag

### Running the Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run the full test suite
python test_format_preservation.py

# Run specific test cases
pytest -xvs test_format_preservation.py::test_all_formats_all_models
```

### Test Scenarios

1. **Format Types**: Bullet lists, numbered lists, tables, and mixed formatting
2. **Language Pairs**: All supported language combinations
3. **Domains**: Healthcare and insurance-specific formatting
4. **Quality Levels**: Performance across different quality settings

### Validation Criteria

The tests validate exact preservation of:

- Line count (with minimal tolerance)
- Bullet point and list formatting
- Numbered list structure
- Table rows and alignment
- Headers and subheaders
- Text styling (bold, italic)

Detailed results are saved to JSON files for analysis and performance tracking.

## System Endpoints

#### GET /supported-languages

Returns available language pairs

#### GET /supported-providers

Returns available providers and models

#### GET /health

Returns system health status and metrics

## Quality Control

### Translation Validation

- Accuracy scoring (35%)

  - Terminology precision
  - Numerical value preservation
  - Abbreviation handling
- Fluency assessment (25%)

  - Natural language flow
  - Domain-appropriate register
  - Grammar and syntax
- Format Preservation (20%)

  - List and bullet point preservation
  - Table structure maintenance
  - Spacing and indentation retention
  - Special formatting elements
- Domain Compliance (20%)

  - Industry standards adherence
  - Regulatory terminology
  - Contextual appropriateness

### Quality Levels

- High: Full validation with detailed feedback
- Medium: Basic validation
- Draft: Quick translation without validation

## Security & Compliance

- Secure API key management
- Input validation and sanitization
- HIPAA-compliant logging
- Rate limiting and error handling
- No sensitive data storage
- Request/response encryption
- Audit trail maintenance

## Error Handling

- Invalid language pair detection
- Domain validation
- Model availability checks
- Rate limit monitoring
- API key validation
- Comprehensive error messages

## Support

For technical support or feature requests:

1. Open a GitHub issue
2. Email: support@example.com
3. Check API documentation at /docs endpoint

## License

MIT License - See LICENSE file for details
