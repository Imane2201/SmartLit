# SmartLit: Advanced Research Assistant with RAG

🧠 **SmartLit** is a comprehensive academic literature analysis platform that combines CrossRef integration, AI-powered analysis, and Retrieval-Augmented Generation (RAG) to help researchers explore, analyze, and synthesize academic literature.

## ✨ Features

### 🔍 **Core Search & Analysis**
- **CrossRef Integration**: Search millions of academic articles from the CrossRef database
- **AI-Powered Analysis**: Automated extraction of research insights using Azure OpenAI
- **Structured Analysis**: Extract objectives, methodology, findings, implications, and limitations
- **Multi-format Support**: PDF upload and processing with text extraction

### 🧠 **RAG-Powered Knowledge Base**
- **Vector Storage**: ChromaDB with Azure OpenAI embeddings for semantic search
- **Document Chunking**: Intelligent text splitting for optimal retrieval
- **Natural Language Queries**: Ask questions about your research collection
- **Semantic Similarity**: Find related articles and concepts

### 📊 **Advanced Analytics**
- **Multi-Article Synthesis**: Generate insights across multiple research papers
- **Research Gap Detection**: AI-powered identification of research opportunities
- **Citation Networks**: Visualize author collaborations, keyword co-occurrences, and article similarities
- **Interactive Dashboards**: Streamlit-based UI with charts and visualizations

### 🔄 **Automation & Monitoring**
- **Article Monitoring**: Automated discovery and analysis of new publications
- **Scheduled Processing**: Regular checks for new articles in your research domains
- **Duplicate Prevention**: Smart tracking of processed articles

### 💾 **Hybrid Storage**
- **Google Sheets Integration**: Structured metadata storage and collaboration
- **Vector Database**: Semantic search and retrieval capabilities
- **Persistent Storage**: Long-term data retention and management

## 🚀 Quick Start

### Prerequisites

1. **Azure OpenAI** account with:
   - Chat completion model (e.g., GPT-4)
   - Text embedding model (e.g., text-embedding-ada-002)

2. **Google Sheets** setup:
   - Service account credentials (`credentials.json`)
   - Google Sheets API enabled

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SmartLit
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (`.env` file):
   ```env
   AZURE_OPENAI_API_KEY=your_azure_openai_key
   AZURE_OPENAI_API_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=your_chat_deployment
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_MODEL_NAME=gpt-4
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment
   AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
   
   SPREADSHEET_ID=your_google_sheets_id
   
   CHROMA_PERSIST_DIRECTORY=./chroma_db
   CHROMA_COLLECTION_NAME=smartlit_articles
   ```

4. **Place Google Service Account credentials**:
   - Put `credentials.json` in the project root

### Running the Application

1. **Start the API server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

2. **Launch the Streamlit dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

3. **Access the application**:
   - API: http://localhost:8000
   - Dashboard: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

## 📚 API Endpoints

### 🔍 **Search & Analysis**

#### `POST /search_articles`
Search and analyze articles from CrossRef with optional filters.

**Request Body:**
```json
{
  "topic": "financial risk management",
  "filters": {
    "year_from": 2020,
    "year_to": 2024,
    "risk_type": "Financial",
    "level_of_analysis": "Firm-level"
  }
}
```

**Response:**
```json
{
  "articles": [...],
  "total_found": 10,
  "topic": "financial risk management",
  "filters_applied": {...}
}
```

### 🧠 **RAG Knowledge Base**

#### `POST /query_knowledge_base`
Query the knowledge base using natural language.

**Request Body:**
```json
{
  "question": "What are the main risk factors in financial institutions?",
  "k": 5,
  "filters": {
    "risk_type": "Financial"
  }
}
```

#### `POST /multi_article_summary`
Generate synthesized insights across multiple articles.

**Request Body:**
```json
{
  "article_titles": [
    "Risk Management in Banks",
    "Financial Risk Assessment"
  ],
  "focus_question": "What are common risk mitigation strategies?"
}
```

#### `GET /suggest_research_gaps`
Analyze patterns and suggest research opportunities.

**Parameters:**
- `domain`: Research domain (default: "risk management")

### 📁 **File Processing**

#### `POST /upload_pdf`
Upload and process PDF research papers.

**Form Data:**
- `file`: PDF file
- `title`: Optional title override
- `authors`: Optional authors (comma-separated)
- `year`: Optional publication year
- `journal`: Optional journal name

### 📊 **Visualization**

#### `POST /generate_citation_graph`
Generate network visualizations of research relationships.

**Parameters:**
- `topic`: Research topic to search
- `graph_type`: "author", "keyword", or "article"
- `max_articles`: Maximum articles to include (default: 20)

**Response includes HTML visualization ready for embedding.**

### 🔄 **Monitoring**

#### `POST /run_monitoring`
Manually run article monitoring for new publications.

#### `GET /monitoring_status`
Get current monitoring configuration and status.

#### `POST /monitoring_config`
Update monitoring settings.

## 🎨 Dashboard Features

The Streamlit dashboard provides an intuitive interface with five main tabs:

### 1. 🔍 **Search Articles**
- Topic-based article search
- Advanced filtering options
- Real-time analysis and visualization
- Interactive charts for risk types and publication years

### 2. 🧠 **Query Knowledge Base**
- Natural language question interface
- Source attribution and relevance scoring
- Filtered search within the knowledge base
- Comprehensive answer generation

### 3. 📊 **Multi-Article Summary**
- Cross-article synthesis
- Focus question customization
- Article selection and analysis
- Structured insight generation

### 4. 🔍 **Research Gaps**
- Domain-specific gap analysis
- Pattern identification
- Opportunity suggestions
- Coverage area mapping

### 5. ℹ️ **About**
- System information and capabilities
- API endpoint documentation
- Feature overview

## 🏗️ Architecture

```
SmartLit Architecture
├── FastAPI Backend
│   ├── CrossRef Integration
│   ├── Azure OpenAI Analysis
│   ├── ChromaDB Vector Store
│   └── Google Sheets Storage
├── Streamlit Frontend
│   ├── Interactive Dashboard
│   ├── Visualization Components
│   └── User Interface
└── Services
    ├── Article Analyzer
    ├── RAG Service
    ├── PDF Processor
    ├── Citation Graph Generator
    └── Article Monitor
```

## 🔧 Configuration

### Azure OpenAI Settings
```python
# Supported models
CHAT_MODELS = ["gpt-4", "gpt-35-turbo"]
EMBEDDING_MODELS = ["text-embedding-ada-002"]

# Recommended settings
TEMPERATURE = 0.3  # For consistent analysis
CHUNK_SIZE = 1000  # For document chunking
CHUNK_OVERLAP = 200  # For context preservation
```

### Vector Store Configuration
```python
# ChromaDB settings
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "smartlit_articles"

# Retrieval settings
DEFAULT_K = 5  # Number of retrieved documents
SIMILARITY_THRESHOLD = 0.7  # For article similarity
```

### Monitoring Configuration
```python
# Default monitoring topics
TOPICS = [
    "financial risk management",
    "operational risk",
    "credit risk assessment",
    "market risk analysis",
    "enterprise risk management"
]

# Monitoring settings
MAX_ARTICLES_PER_TOPIC = 5
MAX_DAYS_SINCE_PUBLICATION = 30
```

## 📊 Data Flow

1. **Article Discovery** → CrossRef API search
2. **Content Analysis** → Azure OpenAI processing
3. **Storage** → Google Sheets (metadata) + ChromaDB (content)
4. **Indexing** → Vector embeddings generation
5. **Retrieval** → Semantic search capabilities
6. **Generation** → RAG-powered insights

## 🔍 Use Cases

### Academic Researchers
- **Literature Reviews**: Automated analysis and synthesis
- **Gap Identification**: Find unexplored research areas
- **Trend Analysis**: Track research evolution over time

### Research Institutions
- **Knowledge Management**: Centralized research repository
- **Collaboration**: Shared insights and findings
- **Monitoring**: Track emerging research trends

### Industry Analysts
- **Market Research**: Academic backing for industry reports
- **Risk Assessment**: Evidence-based risk analysis
- **Competitive Intelligence**: Research landscape mapping

## 🛠️ Customization

### Adding New Analysis Dimensions
1. Update the `ArticleAnalysisSchema` in `langchain_model.py`
2. Modify the analysis prompt
3. Update the Google Sheets headers
4. Adjust the dashboard visualizations

### Custom Vector Store
Replace ChromaDB with alternatives:
- Pinecone
- Weaviate
- FAISS

### Additional Data Sources
Extend beyond CrossRef:
- PubMed integration
- arXiv processing
- Custom database connections

## 🚨 Troubleshooting

### Common Issues

**1. Azure OpenAI Authentication Error**
```
Solution: Verify API key and endpoint in .env file
Check deployment names match your Azure setup
```

**2. Google Sheets Permission Error**
```
Solution: Ensure service account has edit access to the sheet
Verify credentials.json is in the project root
```

**3. ChromaDB Persistence Issues**
```
Solution: Check write permissions for CHROMA_PERSIST_DIRECTORY
Clear the directory if corrupted: rm -rf ./chroma_db
```

**4. PDF Processing Errors**
```
Solution: Ensure PDF contains extractable text (not image-based)
Check file size limits (default: 50MB)
```

## 📈 Performance Optimization

### Vector Store
- **Batch Processing**: Add multiple articles simultaneously
- **Index Optimization**: Regular ChromaDB maintenance
- **Embedding Caching**: Store embeddings to reduce API calls

### API Performance
- **Connection Pooling**: Configure database connections
- **Caching**: Implement Redis for frequent queries
- **Rate Limiting**: Respect API limits for external services

## 🔐 Security Considerations

- **API Keys**: Store securely in environment variables
- **File Uploads**: Validate PDF files before processing
- **Data Privacy**: Consider data retention policies
- **Access Control**: Implement authentication for production use


---

**SmartLit** - Empowering research through intelligent literature analysis 🧠📚 

*Made with ❤️ by Imane LABBASSI*