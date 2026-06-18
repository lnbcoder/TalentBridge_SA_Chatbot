from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL   = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings  = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 10})
llm         = ChatOpenAI(temperature=0, model_name=MODEL)

SYSTEM_PROMPT = """
You are TalentBridge SA's intelligent talent-matching assistant.
TalentBridge SA is a graduate talent development and placement company in South Africa.

You operate in TWO modes:

MODE 1 — TALENT MATCHING & SHORTLISTING
Triggered when the user asks to find, recommend, shortlist, compare, or list
graduates for a role, skill, certification, project, or internship.

HOW TO READ THE PROFILE DATA:
Each graduate profile in the context is structured with these EXACT field labels:

  - "Name:" the graduate's first name (e.g. "Name: Aisha")
  - "Surname:" their surname — but some profiles only have "Name: Aisha Patel"
    with the full name on one line. Read both Name and Surname fields if present,
    or read the full name from the single "Name:" line.
  - "Degree/Qualification:" their degree (e.g. "Bachelor of Science in IT")
  - "Institution:" → their university
  - "Status:" under AVAILABILITY whether they are available or placed
  - "Preferred Role:" → the role they are targeting
  - "Location:" → where they are based

CRITICAL NAME RULES:
- Look for "Name:" in the PERSONAL INFORMATION section of each profile.
- If you see "Name: Aisha Patel" use "Aisha Patel" as the full name.
- If you see "Name: Aisha" and "Surname: Patel" separately, combine them.
- NEVER output "[Name not provided]", "[Graduate 1]", or any placeholder.
- NEVER say a name is missing — it is always in the profile under "Name:".
- NEVER say qualification is missing — it is always under "Degree/Qualification:".

CRITICAL QUALIFICATION RULES:
- The qualification field is labelled "Degree/Qualification:" in the profile.
- Always read the value after "Degree/Qualification:" for the degree name.
- Always read "Institution:" for the university name.
- Format it as: [Degree/Qualification value] — [Institution value]

For EACH suitable graduate output a card in this exact format:

Name: [First name + Surname from the Name field]
Qualification: [value from Degree/Qualification field] — [value from Institution field]
Key Skills: [most relevant skills for the query]
Certifications: [relevant certifications listed in CERTIFICATIONS section]
Experience: [from EXPERIENCE section, or "No formal experience yet" if none]
Relevant Projects: [1–2 projects from PROJECTS BUILT section]
Location: [value from Location field in PERSONAL INFORMATION]
Availability: [value from Status field in AVAILABILITY section]
Why they fit: [1–2 sentence reason based on their skills and qualifications]

After all cards, add a short SUMMARY comparing the candidates.
If a candidate's Status shows "Placed", still list them but note they are currently placed.
If NO graduates match the criteria, say so clearly.


MODE 2 — GENERAL COMPANY / PRODUCT Q&A
Triggered for questions about TalentBridge SA's services, products, culture,
careers, or how the company works.

Answer in a friendly, professional tone using the retrieved context.
Be specific and reference details from the context where possible.
If the context does not contain the answer, say so honestly.

RULES (BOTH MODES)
- ONLY use information found in the retrieved context below.
- NEVER invent names, skills, or certifications not in the context.
- NEVER use placeholders — always use the real values from the profile text.

Retrieved Context:
{context}
"""

TALENT_KEYWORDS = [
    "find graduate", "find graduates", "recommend graduate", "recommend graduates",
    "shortlist", "suitable for", "who has", "who have", "available for",
    "candidates for", "compare", "looking for", "with experience in",
    "with skills in", "with certification", "with qualifications", "hire",
    "intern", "internship", "role", "position", "engineer available",
    "analyst available", "developer available", "designer available",
]

ROLE_EXPANSIONS = {
    "finance":            "accounting financial modelling budgeting IFRS SAIPA CFA Excel",
    "marketing":          "digital marketing SEO social media Google Ads HubSpot Meta campaign",
    "data":               "data science pandas python machine learning Power BI Tableau analytics",
    "software":           "software development programming Python Java JavaScript React Node",
    "cloud":              "AWS Azure cloud DevOps Docker Kubernetes Terraform CI/CD",
    "cybersecurity":      "penetration testing security CompTIA CEH ethical hacking SIEM",
    "civil":              "civil engineering AutoCAD structural design ECSA infrastructure",
    "electrical":         "electrical engineering PLC automation ECSA power systems",
    "business analyst":   "business analysis requirements BPMN JIRA ECBA stakeholder",
    "project management": "project management PMP Scrum Agile PRINCE2 JIRA planning",
    "hr":                 "human resources recruitment labour law SABPP BCEA onboarding",
    "ux":                 "UX UI design Figma user research wireframe prototype usability",
    "mobile":             "Flutter React Native Android iOS mobile development app",
    "ai":                 "artificial intelligence machine learning NLP deep learning TensorFlow PyTorch",
    "graphic design":     "graphic design Adobe Illustrator Photoshop branding motion",
    "erp":                "SAP ERP Odoo systems implementation information systems",
    "network":            "networking CCNA CCNP Cisco routing switching VPN SD-WAN",
    "accounting":         "accounting SAIPA IFRS financial statements audit Sage Xero",
}


def _is_talent_query(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in TALENT_KEYWORDS)


def _enrich_query(query: str) -> str:
    """Append domain keywords so vector search retrieves the right profiles."""
    q_lower = query.lower()
    expansions = [keyword for role, keyword in ROLE_EXPANSIONS.items() if role in q_lower]
    if expansions:
        return f"{query}\n\nRelevant skills and keywords: {' '.join(expansions)}"
    return query


# ── Core functions ─────────────────────────────────────────────────────────────

def resolve_query(query: str, history: list) -> str:
    last_user = None
    if history:
        for turn in reversed(history):
            if turn.get("role") == "user":
                last_user = turn["content"]
                break
    combined = f"{last_user}\n{query}" if last_user else query
    return _enrich_query(combined) if _is_talent_query(combined) else combined


def fetch_context(query: str) -> list[Document]:
    """Retrieve top-k relevant chunks from the vectorstore."""
    return retriever.invoke(query)


def answer_question(query: str, history=None):
    history = history or []

    resolved_query = resolve_query(query, history)
    docs = fetch_context(resolved_query)

    # Include source filename in context so LLM has name as extra signal
    context_parts = []
    for doc in docs:
        source   = doc.metadata.get("source", "")
        doc_type = doc.metadata.get("doc_type", "")
        filename = Path(source).stem   # e.g. "aisha patel" from "aisha patel.md"
        header   = f"[File: {filename} | Category: {doc_type}]"
        context_parts.append(f"{header}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    for turn in history:
        role    = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query))

    response = llm.invoke(messages)
    return response.content, docs