"""
jd_parser.py — Job Description Parser
Extracts structured information from raw JD text.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

JD_PARSE_TEMPLATE = """Extract the following fields from the job description below.
Return each field on its own line in the format: FIELD_NAME: value

Fields:
- JOB_TITLE
- COMPANY
- EXPERIENCE
- TECHNICAL_SKILLS
- SOFT_SKILLS
- RESPONSIBILITIES

Job Description:
{jd_text}
"""


def parse_jd(jd_text: str) -> dict:
    """Parse a job description into structured fields."""
    prompt = PromptTemplate(
        input_variables=["jd_text"],
        template=JD_PARSE_TEMPLATE
    )
    chain = prompt | llm
    response = chain.invoke({"jd_text": jd_text[:3000]})

    result = {}
    for line in response.content.strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()

    return result
