1. Project Objective and Use Case
Project Objective
The objective of this project is to develop an intelligent TalentBridge SA assistant that helps users:
  •	Search and identify graduate candidates based on skills, qualifications, certifications, and experience.
  •	Generate candidate shortlists for job opportunities and internships.
  •	Answer questions about TalentBridge SA services and offerings.
  •	Improve recruitment efficiency through AI-powered knowledge retrieval and recommendation capabilities.

Use Case
The chatbot serves two primary functions:
  Talent Matching and Candidate Shortlisting
  Users can ask questions such as:
    •	"Find graduates suitable for a finance internship."
    •	"Who has Power BI and accounting experience?"
    •	"Create a shortlist for an HR Administrator role."
  The system retrieves relevant graduate profiles and generates structured recommendations.
  
Company Knowledge Assistant
  Users can ask questions such as:
    •	"What services does TalentBridge SA provide?"
    •	"How does the graduate placement process work?"
  The system retrieves company information and generates context-aware responses. 

3. Dataset and Data Sources
Dataset Description
  The dataset consists of Markdown (.md) documents stored in the project's knowledge-base directory.
  The documents contain:
    Graduate Profiles
      Information such as:
        •	Name
        •	Qualification
        •	Institution
        •	Skills
        •	Certifications
        •	Projects
        •	Work experience
        •	Availability
        •	Preferred role
        •	Location

TalentBridge Company Information
  Business information including:
    •	Services offered
    •	Talent development programmes
    •	Placement solutions
    •	Recruitment offerings

Data Source
  Local knowledge base:
  
knowledge-base/
    company/ 
        adout.md
        careers.md
        culture.md
    employee/
        profile1.md
        profile2.md
        ...
    product/
        products_and_services.md


3. RAG Architecture and Workflow
Architecture Overview
 Knowledge Base (Markdown Files) -> Document Loader -> Text Splitter -> OpenAI Embeddings -> Pinecone Vector DB -> Retriever -> GPT-4.1 Nano -> TalentBridge Assistant

Workflow
Step 1: Data Ingestion
  The system loads Markdown documents from the knowledge base

Step 2: Chunking
  Documents are split into smaller chunks 

Step 3: Embedding Generation
  Each text chunk is converted into vector embeddings

Step 4: Vector Storage
  Embeddings are stored in a Pinecone vector database.

Step 5: Retrieval
  User queries are converted into embeddings.

Step 6: Response Generation
  Retrieved context is sent to GPT-4.1 Nano through LangChain.

Step 7: User Interface
  Responses are displayed in a Gradio web application.


 
4. Tools, Frameworks and Technologies Used
  Frontend: Gradio 
  Backend: Python, LangChain 
  LLM:  OpenAI GPT-4.1 Nano 
  Embeddings: OpenAI text-embedding-3-large 
  Vector Database: Pinecone 
  Knowledge Base:  Markdown Documents 
  Deployment:  Hugging Face Spaces 
  Development: VS Code, GitHub, UV 
  Configuration: Python Dotenv 
