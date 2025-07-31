import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import json
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="SmartLit Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL (adjust as needed)
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .article-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #ffffff;
    }
    .source-card {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API calls to the SmartLit backend"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {}

def display_knowledge_base_stats():
    """Display knowledge base statistics in the sidebar"""
    with st.sidebar:
        st.subheader("📊 Knowledge Base Stats")
        stats = call_api("/knowledge_base_stats")
        
        if stats:
            st.metric("Total Documents", stats.get("total_documents", 0))
            st.metric("Collection", stats.get("collection_name", "N/A"))
        else:
            st.warning("Could not load stats")

def search_articles_tab():
    """Article search and analysis tab"""
    st.markdown('<div class="section-header">🔍 Search & Analyze Articles</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input("Research Topic", placeholder="e.g., 'financial risk management'")
        
    with col2:
        search_button = st.button("🔍 Search Articles", type="primary")
    
    # Advanced filters
    with st.expander("🎛️ Advanced Filters"):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            year_from = st.number_input("Year From", min_value=1990, max_value=2024, value=2020)
            year_to = st.number_input("Year To", min_value=1990, max_value=2024, value=2024)
        
        with filter_col2:
            risk_type = st.selectbox("Risk Type", ["", "Financial", "Operational", "Strategic", "Credit", "Market"])
            level_of_analysis = st.selectbox("Level of Analysis", ["", "Firm-level", "Industry-level", "Country-level", "Multi-level"])
        
        with filter_col3:
            journal = st.text_input("Journal (optional)")
    
    if search_button and topic:
        with st.spinner("Searching and analyzing articles..."):
            # Prepare filters
            filters = {}
            if year_from:
                filters["year_from"] = year_from
            if year_to:
                filters["year_to"] = year_to
            if risk_type:
                filters["risk_type"] = risk_type
            if level_of_analysis:
                filters["level_of_analysis"] = level_of_analysis
            if journal:
                filters["journal"] = journal
            
            # Call API
            data = {"topic": topic}
            if filters:
                data["filters"] = filters
                
            results = call_api("/search_articles", method="POST", data={"topic": topic, "filters": filters if filters else None})
            
            if results and "articles" in results:
                articles = results["articles"]
                st.success(f"Found {len(articles)} articles")
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Articles Found", len(articles))
                
                with col2:
                    risk_types = [a.get("risk_type", "Unknown") for a in articles if a.get("risk_type")]
                    st.metric("Risk Types", len(set(risk_types)))
                
                with col3:
                    years = [a.get("year") for a in articles if a.get("year")]
                    avg_year = sum(years) / len(years) if years else 0
                    st.metric("Avg Year", f"{avg_year:.0f}" if avg_year else "N/A")
                
                with col4:
                    journals = [a.get("journal") for a in articles if a.get("journal")]
                    st.metric("Unique Journals", len(set(journals)))
                
                # Visualizations
                if articles:
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Risk type distribution
                        risk_counts = pd.Series(risk_types).value_counts()
                        if not risk_counts.empty:
                            fig_risk = px.bar(
                                x=risk_counts.values,
                                y=risk_counts.index,
                                orientation='h',
                                title="Risk Type Distribution",
                                labels={'x': 'Count', 'y': 'Risk Type'}
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)
                    
                    with viz_col2:
                        # Year distribution
                        if years:
                            year_counts = pd.Series(years).value_counts().sort_index()
                            fig_year = px.line(
                                x=year_counts.index,
                                y=year_counts.values,
                                title="Publications by Year",
                                labels={'x': 'Year', 'y': 'Count'}
                            )
                            st.plotly_chart(fig_year, use_container_width=True)
                
                # Display articles
                st.subheader("📄 Article Details")
                for i, article in enumerate(articles):
                    with st.expander(f"📑 {article.get('title', 'Unknown Title')}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Authors:** {', '.join(article.get('authors', []))}")
                            st.write(f"**Journal:** {article.get('journal', 'Unknown')}")
                            st.write(f"**Year:** {article.get('year', 'Unknown')}")
                            st.write(f"**Risk Type:** {article.get('risk_type', 'Unknown')}")
                            st.write(f"**Level of Analysis:** {article.get('level_of_analysis', 'Unknown')}")
                        
                        with col2:
                            st.write(f"**Objective:** {article.get('objective', 'N/A')[:200]}...")
                            st.write(f"**Main Findings:** {article.get('main_findings', 'N/A')[:200]}...")
                        
                        if st.button(f"View Full Analysis {i}", key=f"analysis_{i}"):
                            st.json(article)

def rag_query_tab():
    """RAG knowledge base query tab"""
    st.markdown('<div class="section-header">🧠 Query Knowledge Base</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Ask questions about the research articles in the knowledge base. The system will find relevant 
    articles and provide comprehensive answers based on the research findings.
    """)
    
    # Query input
    question = st.text_area(
        "Your Question",
        placeholder="e.g., 'What are the main risk factors identified in financial institutions?'"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        k_results = st.slider("Number of sources", min_value=3, max_value=10, value=5)
    
    with col2:
        query_button = st.button("🔍 Query Knowledge Base", type="primary")
    
    # Query filters
    with st.expander("🎛️ Query Filters"):
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            filter_risk_type = st.selectbox("Filter by Risk Type", ["", "Financial", "Operational", "Strategic"])
            filter_year = st.number_input("Filter by Year", min_value=1990, max_value=2024, value=None)
        
        with filter_col2:
            filter_journal = st.text_input("Filter by Journal")
    
    if query_button and question:
        with st.spinner("Searching knowledge base and generating answer..."):
            # Prepare filters
            filters = {}
            if filter_risk_type:
                filters["risk_type"] = filter_risk_type
            if filter_year:
                filters["year"] = filter_year
            if filter_journal:
                filters["journal"] = filter_journal
            
            query_data = {
                "question": question,
                "k": k_results,
                "filters": filters if filters else None
            }
            
            result = call_api("/query_knowledge_base", method="POST", data=query_data)
            
            if result and "answer" in result:
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(result["answer"])
                
                # Display sources
                st.markdown("### 📚 Sources")
                sources = result.get("sources", [])
                
                for i, source in enumerate(sources):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source {i+1}:</strong> {source.get('title', 'Unknown')}<br>
                        <strong>Authors:</strong> {source.get('authors', 'Unknown')}<br>
                        <strong>Year:</strong> {source.get('year', 'Unknown')} | 
                        <strong>Journal:</strong> {source.get('journal', 'Unknown')}<br>
                        <strong>Content:</strong> {source.get('chunk_content', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info(f"Answer based on {len(sources)} relevant sources")

def multi_article_summary_tab():
    """Multi-article summary generation tab"""
    st.markdown('<div class="section-header">📊 Multi-Article Summary</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate synthesized insights across multiple specific research articles. 
    Enter the titles of articles you want to analyze together.
    """)
    
    # Article titles input
    article_titles_text = st.text_area(
        "Article Titles (one per line)",
        placeholder="Enter article titles, one per line..."
    )
    
    focus_question = st.text_input(
        "Focus Question (optional)",
        placeholder="e.g., 'What are the common risk mitigation strategies?'"
    )
    
    if st.button("🔄 Generate Summary", type="primary"):
        if article_titles_text:
            article_titles = [title.strip() for title in article_titles_text.split('\n') if title.strip()]
            
            with st.spinner("Analyzing articles and generating synthesis..."):
                summary_data = {
                    "article_titles": article_titles,
                    "focus_question": focus_question if focus_question else None
                }
                
                result = call_api("/multi_article_summary", method="POST", data=summary_data)
                
                if result and "summary" in result:
                    # Display summary
                    st.markdown("### 📋 Synthesis")
                    st.markdown(result["summary"])
                    
                    # Display articles analyzed
                    st.markdown("### 📚 Articles Analyzed")
                    articles_analyzed = result.get("articles_analyzed", [])
                    
                    for article in articles_analyzed:
                        st.markdown(f"""
                        - **{article.get('title', 'Unknown')}** 
                          - Authors: {article.get('authors', 'Unknown')}
                          - Year: {article.get('year', 'Unknown')}
                          - Journal: {article.get('journal', 'Unknown')}
                        """)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Articles Found", result.get("articles_found", 0))
                    with col2:
                        st.metric("Chunks Analyzed", result.get("total_chunks_analyzed", 0))
                    with col3:
                        if result.get("focus_question"):
                            st.info(f"Focus: {result['focus_question']}")

def research_gaps_tab():
    """Research gaps analysis tab"""
    st.markdown('<div class="section-header">🔍 Research Gap Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze the knowledge base to identify potential research gaps and opportunities 
    in your domain of interest.
    """)
    
    domain = st.text_input(
        "Research Domain",
        value="risk management",
        placeholder="e.g., 'financial risk management', 'cybersecurity', 'supply chain'"
    )
    
    if st.button("🔍 Analyze Research Gaps", type="primary"):
        if domain:
            with st.spinner("Analyzing research patterns and identifying gaps..."):
                result = call_api(f"/suggest_research_gaps?domain={domain}")
                
                if result and "gap_analysis" in result:
                    # Display gap analysis
                    st.markdown("### 🔍 Research Gap Analysis")
                    st.markdown(result["gap_analysis"])
                    
                    # Display metadata
                    st.markdown("### 📊 Analysis Details")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Articles Analyzed", result.get("articles_analyzed", 0))
                        st.metric("Domain", result.get("domain", "Unknown"))
                    
                    with col2:
                        coverage_areas = result.get("coverage_areas", [])
                        if coverage_areas:
                            st.write("**Coverage Areas:**")
                            for area in coverage_areas:
                                if area:  # Filter out None/empty values
                                    st.write(f"• {area}")

def main():
    """Main dashboard function"""
    st.markdown('<div class="main-header">🧠 SmartLit Research Dashboard</div>', unsafe_allow_html=True)
    
    # Display knowledge base stats in sidebar
    display_knowledge_base_stats()
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Search Articles", 
        "🧠 Query Knowledge Base", 
        "📊 Multi-Article Summary",
        "🔍 Research Gaps",
        "ℹ️ About"
    ])
    
    with tab1:
        search_articles_tab()
    
    with tab2:
        rag_query_tab()
    
    with tab3:
        multi_article_summary_tab()
    
    with tab4:
        research_gaps_tab()
    
    with tab5:
        st.markdown("""
        ## About SmartLit
        
        SmartLit is an advanced research assistant that combines:
        
        - **CrossRef Integration**: Search academic articles from a vast database
        - **AI-Powered Analysis**: Automated extraction of research insights using Azure OpenAI
        - **RAG Technology**: Query your knowledge base with natural language
        - **Multi-Article Synthesis**: Generate insights across multiple research papers
        - **Research Gap Detection**: Identify opportunities in your research domain
        
        ### Features:
        - 🔍 **Semantic Search**: Find relevant articles using natural language
        - 🧠 **Knowledge Base Queries**: Ask questions about your research collection
        - 📊 **Visual Analytics**: Interactive charts and metrics
        - 📚 **Multi-Paper Analysis**: Cross-paper synthesis and comparison
        - 🎯 **Gap Analysis**: Automated research opportunity identification
        
        ### API Endpoints:
        - `/search_articles` - Search and analyze new articles
        - `/query_knowledge_base` - RAG-powered Q&A
        - `/multi_article_summary` - Cross-article synthesis
        - `/suggest_research_gaps` - Gap analysis
        - `/search_similar` - Semantic similarity search
        
        Built with FastAPI, LangChain, ChromaDB, and Streamlit.
        """)

if __name__ == "__main__":
    main()