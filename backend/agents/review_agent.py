
"""
Code Review Agent for AI DevTeam / AutoDev.
Reviews generated code using Gemini and RAG guidelines.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from backend.schemas.state import AgentState, SecurityReviewReport
from backend.core.retriever import retrieve_guidelines
from backend.config import config


def run_review_agent(state: AgentState) -> AgentState:
    """
    Reviews generated code and updates AgentState
    with a structured security review report.
    """

    state["current_agent"] = "Code Review Agent"
    state["status"] = "reviewing"

    if "logs" not in state or state["logs"] is None:
        state["logs"] = []

    state["logs"].append(
        "[Review Agent] Starting security and code quality review..."
    )

    # Get files to review
    files = state.get("patched_files") or state.get("generated_files") or []

    if not files:
        state["logs"].append(
            "[Review Agent] No generated files available for review."
        )
        state["security_report"] = SecurityReviewReport(
            quality_score=0,
            owasp_compliance=False,
            vulnerabilities=["No generated code was provided for review."],
            suggestions=["Provide generated files before running code review."]
        ).model_dump()
        return state

    # Prepare code content
    code_content = ""

    for file in files:
        code_content += (
            f"\n\nFile: {file.get('path', 'unknown')}\n"
            f"Language: {file.get('language', 'unknown')}\n"
            f"Code:\n{file.get('content', '')}\n"
        )

    # Retrieve relevant RAG guidelines
    try:
        guidelines = retrieve_guidelines(
            "Security vulnerabilities, secure API development, "
            "Python coding standards and code quality"
        )

        guideline_context = "\n".join(
            [
                f"Category: {item['metadata']['category']}\n"
                f"Guideline: {item['document']}"
                for item in guidelines
            ]
        )

    except Exception as e:
        guideline_context = ""
        state["logs"].append(
            f"[Review Agent] RAG retrieval warning: {str(e)}"
        )

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.2,
    )

    # Structured output parser
    parser = PydanticOutputParser(
        pydantic_object=SecurityReviewReport
    )

    # Review prompt
    prompt_template = PromptTemplate(
        template="""You are a senior software security engineer.

Review the following generated source code for security,
OWASP-related risks, and code quality.

CODE TO REVIEW:
{code_content}

RETRIEVED CODING AND SECURITY GUIDELINES:
{guideline_context}

Instructions:
1. Identify actual security vulnerabilities visible in the code.
2. Do not invent vulnerabilities without evidence.
3. Check input validation, authentication, authorization,
   error handling and sensitive data exposure where relevant.
4. Give practical improvement suggestions.
5. Assign a quality score between 0 and 100.
6. Set owasp_compliance to false if you identify an
   OWASP-related security violation. Do not claim complete
   OWASP compliance from this limited review alone.
7. Clearly distinguish confirmed issues from potential risks.

{format_instructions}
""",
        input_variables=["code_content", "guideline_context"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    try:
        formatted_prompt = prompt_template.format(
            code_content=code_content,
            guideline_context=guideline_context or "No guidelines retrieved."
        )

        response = llm.invoke(formatted_prompt)

        if isinstance(response.content, str):
            response_text = response.content
        else:
            response_text = "\n".join(
                block["text"]
                for block in response.content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        report = parser.parse(response_text)

        state["security_report"] = report.model_dump()

        state["logs"].append(
            f"[Review Agent] Review completed. Quality score: "
            f"{report.quality_score}/100"
        )

    except Exception as e:
        state["logs"].append(
            f"[Review Agent Error] Review failed: {str(e)}"
        )
        raise

    return state
