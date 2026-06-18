SECTION 1 — PROJECT OBJECTIVE & USE CASE
────────────────────────────────────────────────────

1.1 PROJECT OBJECTIVE
----------------------
Build a Retrieval-Augmented Generation (RAG) conversational assistant that
allows companies and HR professionals to query TalentBridge SA's graduate
talent pool using natural language, and receive intelligent, structured
candidate recommendations backed by real profile data.

The assistant must be able to:
  - Match graduates to job roles and internship briefs
  - Filter candidates by skills, qualifications, certifications, experience,
    location, and availability
  - Generate structured shortlists with comparative summaries
  - Answer general questions about TalentBridge SA products and services
  - Maintain conversation context across multi-turn dialogue

1.2 USE CASES
--------------
The following are the primary use cases the system must handle:

USE CASE 1 — Talent Search by Role
  Example: "Find graduates suitable for a finance internship."
  Expected: System retrieves finance-related graduate profiles and ranks them
  by relevance to finance internship requirements.

USE CASE 2 — Skill-Based Search
  Example: "Who has experience with Power BI and Accounting?"
  Expected: System identifies graduates listing Power BI and accounting skills.

USE CASE 3 — Certification Filter
  Example: "Recommend marketing graduates with digital marketing certifications."
  Expected: Returns marketing graduates holding Google, HubSpot, or Meta certs.

USE CASE 4 — Availability Search
  Example: "Find civil engineers available immediately."
  Expected: Returns civil/electrical engineering graduates with status
  "Immediately available."

USE CASE 5 — Skill Domain Query
  Example: "Which graduates have project management skills?"
  Expected: Returns all graduates with PM skills, tools, or certifications.

USE CASE 6 — Role Shortlisting
  Example: "Create a shortlist for an HR Administrator role."
  Expected: Structured candidate cards ranked by HR suitability.

USE CASE 7 — Product Query
  Example: "What TalentBridge service helps companies build project teams?"
  Expected: Explains the BuildLab product, how it works, and pricing.

USE CASE 8 — Candidate Comparison
  Example: "Compare available candidates for a Business Analyst position."
  Expected: Side-by-side comparison of BA graduates' skills and suitability.

2.1 DATA OVERVIEW
------------------
The knowledge base consists of structured text documents (.md files)
organised into three categories (folders):

FOLDER STRUCTURE:
  rag_project/
  ├── company/
  │   ├── about.md            — Company mission, history, values, impact
  │   ├── overview.md         — Business model, revenue, partnerships, metrics
  │   ├── culture.md          — Work environment, DEI, wellness, recognition
  │   └── careers.md          — How graduates join, open positions, benefits
  │
  ├── employees/
  │   ├── employee_01_sipho_nkosi.md
  │   ├── employee_02_naledi_dlamini.md
  │   ├── ... (20 graduate profiles total)
  │   └── employee_20_lindiwe_ntuli.md
  │
  └── products/
      └── products_and_services.md  — TalentConnect, BuildLab, SkillBridge

2.2 GRADUATE PROFILE SCHEMA
-----------------------------
Each employee .md file follows a consistent schema:

  - Personal Information (Name, Surname, DOB, Location, Contact)
  - Qualification (Degree, Institution, Year, NQF Level, Subjects)
  - Skills (Technical + Tools + Soft skills)
  - Certifications (Name, Issuer, Year)
  - Experience (Company, Role, Duration, Responsibilities)
  - Projects Built (Name, Description, Tech Stack)
  - Achievements (Awards, Rankings, Publications)
  - Availability (Status, Preferred Role, Location, Notice Period)

2.3 DATASET STATISTICS
------------------------
  - Total documents: 25 md files
  - Company documents: 4
  - Employee profiles: 20
  - Product documents: 1
  - Average profile length: ~600–900 words
  - Total estimated tokens in knowledge base: ~40,000


3.1 HIGH-LEVEL ARCHITECTURE
-----------------------------

  [User Query]
       │
       V
  [Gradio Chat UI — app.py]
       │
       V
  [answer.py — Query Resolver + Enricher]
       │
       │  Enriches talent-matching queries with domain keywords
       │  Prepends prior conversation turn for follow-up questions
       V
  [OpenAI text-embedding-3-large — Embed Query]
       │
       V
  [ChromaDB Vector Store — Cosine Similarity Search (top-k=10)]
       │
       V
  [Retrieved Context Documents (chunks)]
       │
       V
  [answer.py — Prompt Assembly]
       │  System Prompt (talent-matching or general mode)
       │  + Conversation History
       │  + Retrieved Context
       │  + User Query
       V
  [OpenAI GPT-4.1-nano — LLM Response Generation]
       │
       V
  [Structured Answer + Source Citations]
       │
       V
  [Gradio Chat UI — Display Answer + Retrieved Context Panel]


3.2 WORKFLOW — STEP BY STEP (Runtime)
---------------------------------------
Step 1: User types a question in the Gradio chat UI
Step 2: app.py calls answer_question(message, history)
Step 3: answer.py checks if query is a talent-matching query
Step 4: If yes, enriches the query with role-specific keywords
Step 5: If history exists, prepends last user message for context continuity
Step 6: Enriched query is embedded using text-embedding-3-large
Step 7: ChromaDB returns top-10 most similar document chunks
Step 8: Chunks are concatenated into a context string
Step 9: System prompt (Mode 1 or 2) is assembled with context injected
Step 10: Full message list (system + history + new query) sent to GPT-4.1-nano
Step 11: LLM generates a structured response
Step 12: app.py displays response in chat window
Step 13: Retrieved source chunks displayed in the right-hand context panel
Step 14: Conversation history updated for the next turn
