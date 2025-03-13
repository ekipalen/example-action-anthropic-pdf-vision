import base64
import os

import anthropic
from dotenv import load_dotenv
from sema4ai.actions import ActionError, Response, Secret, action

load_dotenv()

MODEL = "claude-3-7-sonnet-20250219"


@action
def extract_data_from_pdf(pdf_path: str, prompt: str, api_key: Secret) -> Response:
    """Extracts data from a PDF file using Anthropic.

    Args:
        pdf_path: Path to the local PDF file
        prompt: The prompt to guide the extraction process
        api_key: The API key for Anthropic
    Returns:
        Response: A Response object containing extracted text in a serializable dictionary
    Raises:
        ActionError: If file not found, API error, or other processing errors occur
    """
    try:
        # Load and encode the local PDF
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")

        key = api_key.value or os.getenv("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        result = {
            "content": [{"text": message.content[0].text, "type": "text"}],
            "status": "success",
        }

        return Response(result=result)

    except FileNotFoundError:
        raise ActionError(f"File not found: {pdf_path}")
    except anthropic.APIError as e:
        raise ActionError(f"Anthropic API error: {str(e)}")
    except Exception as e:
        raise ActionError(f"Unexpected error: {str(e)}")
