import networkx as nx
from pyvis.network import Network
import json
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import re

from .vector_store import VectorStoreService


class CitationGraphGenerator:
    def __init__(self):
        """Initialize the citation graph generator with vector store access"""
        self.vector_store = VectorStoreService()
    
    def create_author_network(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a network graph of author collaborations
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary containing network data and statistics
        """
        G = nx.Graph()
        
        # Track author information
        author_articles = defaultdict(list)
        author_info = {}
        
        for article in articles:
            authors = article.get('authors', [])
            if isinstance(authors, str):
                authors = [authors]
            
            # Clean author names
            clean_authors = [self._clean_author_name(author) for author in authors if author]
            
            if len(clean_authors) > 1:
                # Add authors as nodes
                for author in clean_authors:
                    if author not in author_info:
                        author_info[author] = {
                            'articles': 0,
                            'risk_types': set(),
                            'years': []
                        }
                    
                    author_info[author]['articles'] += 1
                    if article.get('risk_type'):
                        author_info[author]['risk_types'].add(article['risk_type'])
                    if article.get('year'):
                        author_info[author]['years'].append(article['year'])
                    
                    author_articles[author].append(article.get('title', 'Unknown'))
                
                # Add collaboration edges
                for i, author1 in enumerate(clean_authors):
                    for author2 in clean_authors[i+1:]:
                        if G.has_edge(author1, author2):
                            G[author1][author2]['weight'] += 1
                            G[author1][author2]['collaborations'].append(article.get('title', 'Unknown'))
                        else:
                            G.add_edge(author1, author2, weight=1, 
                                     collaborations=[article.get('title', 'Unknown')])
        
        # Add node attributes
        for author, info in author_info.items():
            G.add_node(author, 
                      articles=info['articles'],
                      risk_types=list(info['risk_types']),
                      years=info['years'],
                      avg_year=sum(info['years']) / len(info['years']) if info['years'] else 0)
        
        # Calculate network metrics
        stats = self._calculate_network_stats(G)
        
        return {
            'network': G,
            'stats': stats,
            'author_info': author_info,
            'author_articles': dict(author_articles)
        }
    
    def create_keyword_network(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a network graph of keyword co-occurrences
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary containing network data and statistics
        """
        G = nx.Graph()
        
        # Extract keywords from various fields
        all_keywords = []
        keyword_articles = defaultdict(list)
        
        for article in articles:
            article_keywords = set()
            
            # Extract from various text fields
            text_fields = ['title', 'abstract', 'objective', 'key_variables', 'main_findings']
            combined_text = ' '.join([str(article.get(field, '')) for field in text_fields])
            
            # Extract meaningful keywords (simple approach)
            keywords = self._extract_keywords(combined_text)
            
            # Add risk type as keyword
            if article.get('risk_type'):
                keywords.add(article['risk_type'].lower())
            
            # Add level of analysis as keyword
            if article.get('level_of_analysis'):
                keywords.add(article['level_of_analysis'].lower().replace('-', ' '))
            
            all_keywords.extend(keywords)
            
            # Track which articles contain each keyword
            for keyword in keywords:
                keyword_articles[keyword].append(article.get('title', 'Unknown'))
                article_keywords.add(keyword)
            
            # Add co-occurrence edges
            keywords_list = list(article_keywords)
            for i, kw1 in enumerate(keywords_list):
                for kw2 in keywords_list[i+1:]:
                    if G.has_edge(kw1, kw2):
                        G[kw1][kw2]['weight'] += 1
                        G[kw1][kw2]['articles'].append(article.get('title', 'Unknown'))
                    else:
                        G.add_edge(kw1, kw2, weight=1, 
                                 articles=[article.get('title', 'Unknown')])
        
        # Add node attributes (keyword frequency)
        keyword_counts = Counter(all_keywords)
        for keyword, count in keyword_counts.items():
            G.add_node(keyword, frequency=count, articles=keyword_articles[keyword])
        
        # Filter to keep only keywords that appear in multiple articles
        nodes_to_remove = [node for node, data in G.nodes(data=True) 
                          if data.get('frequency', 0) < 2]
        G.remove_nodes_from(nodes_to_remove)
        
        stats = self._calculate_network_stats(G)
        
        return {
            'network': G,
            'stats': stats,
            'keyword_counts': keyword_counts,
            'keyword_articles': dict(keyword_articles)
        }
    
    def create_article_similarity_network(self, articles: List[Dict[str, Any]], 
                                        similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Create a network based on article similarity using semantic embeddings
        
        Args:
            articles: List of article dictionaries
            similarity_threshold: Minimum similarity to create an edge
            
        Returns:
            Dictionary containing network data and statistics
        """
        G = nx.Graph()
        
        # Create simplified network based on shared attributes for now
        # In a production system, you'd use actual semantic similarity
        
        article_data = {}
        
        for i, article in enumerate(articles):
            article_id = f"article_{i}"
            title = article.get('title', f'Article {i}')
            
            # Add article as node
            G.add_node(article_id, 
                      title=title,
                      authors=article.get('authors', []),
                      year=article.get('year'),
                      risk_type=article.get('risk_type', ''),
                      level_of_analysis=article.get('level_of_analysis', ''))
            
            article_data[article_id] = article
        
        # Create edges based on shared attributes
        nodes = list(G.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                similarity_score = self._calculate_article_similarity(
                    article_data[node1], article_data[node2]
                )
                
                if similarity_score >= similarity_threshold:
                    G.add_edge(node1, node2, 
                             weight=similarity_score,
                             similarity=similarity_score)
        
        stats = self._calculate_network_stats(G)
        
        return {
            'network': G,
            'stats': stats,
            'article_data': article_data
        }
    
    def generate_html_visualization(self, network_data: Dict[str, Any], 
                                  network_type: str = "author") -> str:
        """
        Generate HTML visualization using PyVis
        
        Args:
            network_data: Network data from create_*_network methods
            network_type: Type of network ('author', 'keyword', 'article')
            
        Returns:
            HTML string for the visualization
        """
        G = network_data['network']
        
        # Create PyVis network
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        # Add nodes with styling based on network type
        for node, data in G.nodes(data=True):
            if network_type == "author":
                size = min(50, 10 + data.get('articles', 1) * 5)
                title = f"Author: {node}<br>Articles: {data.get('articles', 0)}<br>Risk Types: {', '.join(data.get('risk_types', []))}"
                color = self._get_color_by_risk_type(data.get('risk_types', ['Unknown'])[0] if data.get('risk_types') else 'Unknown')
            
            elif network_type == "keyword":
                size = min(50, 10 + data.get('frequency', 1) * 3)
                title = f"Keyword: {node}<br>Frequency: {data.get('frequency', 0)}<br>Articles: {len(data.get('articles', []))}"
                color = self._get_color_by_frequency(data.get('frequency', 1))
            
            elif network_type == "article":
                size = 20
                title = f"Title: {data.get('title', node)}<br>Year: {data.get('year', 'Unknown')}<br>Risk Type: {data.get('risk_type', 'Unknown')}"
                color = self._get_color_by_risk_type(data.get('risk_type', 'Unknown'))
            
            net.add_node(node, label=self._truncate_label(str(node)), 
                        size=size, title=title, color=color)
        
        # Add edges
        for edge in G.edges(data=True):
            weight = edge[2].get('weight', 1)
            width = min(10, weight * 2)
            
            if network_type == "author":
                title = f"Collaborations: {weight}<br>Papers: {', '.join(edge[2].get('collaborations', [])[:3])}"
            elif network_type == "keyword":
                title = f"Co-occurrences: {weight}<br>Shared articles: {len(edge[2].get('articles', []))}"
            elif network_type == "article":
                title = f"Similarity: {edge[2].get('similarity', 0):.2f}"
            
            net.add_edge(edge[0], edge[1], width=width, title=title)
        
        # Generate HTML
        html_string = net.generate_html()
        
        return html_string
    
    def _clean_author_name(self, author_name: str) -> str:
        """Clean and normalize author names"""
        if not author_name or not isinstance(author_name, str):
            return ""
        
        # Remove extra whitespace and common prefixes/suffixes
        name = re.sub(r'\s+', ' ', author_name.strip())
        name = re.sub(r'^(Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?)\s+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+(Jr\.?|Sr\.?|PhD\.?|MD\.?)$', '', name, flags=re.IGNORECASE)
        
        return name.title()
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> set:
        """Extract meaningful keywords from text"""
        if not text:
            return set()
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'cannot', 'from', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
        
        keywords = set()
        for word in words:
            if (len(word) >= min_length and 
                word not in stop_words and 
                not word.isdigit() and
                word.isalpha()):
                keywords.add(word)
        
        return keywords
    
    def _calculate_article_similarity(self, article1: Dict, article2: Dict) -> float:
        """Calculate similarity between two articles based on attributes"""
        similarity = 0.0
        total_weight = 0.0
        
        # Risk type similarity
        if article1.get('risk_type') and article2.get('risk_type'):
            if article1['risk_type'] == article2['risk_type']:
                similarity += 0.3
            total_weight += 0.3
        
        # Level of analysis similarity
        if article1.get('level_of_analysis') and article2.get('level_of_analysis'):
            if article1['level_of_analysis'] == article2['level_of_analysis']:
                similarity += 0.2
            total_weight += 0.2
        
        # Year similarity (within 3 years)
        if article1.get('year') and article2.get('year'):
            year_diff = abs(article1['year'] - article2['year'])
            if year_diff <= 3:
                similarity += 0.1 * (1 - year_diff / 3)
            total_weight += 0.1
        
        # Author overlap
        authors1 = set(article1.get('authors', []))
        authors2 = set(article2.get('authors', []))
        if authors1 and authors2:
            overlap = len(authors1.intersection(authors2))
            if overlap > 0:
                similarity += 0.4 * (overlap / max(len(authors1), len(authors2)))
            total_weight += 0.4
        
        return similarity / total_weight if total_weight > 0 else 0.0
    
    def _calculate_network_stats(self, G: nx.Graph) -> Dict[str, Any]:
        """Calculate network statistics"""
        if len(G.nodes()) == 0:
            return {"nodes": 0, "edges": 0}
        
        stats = {
            "nodes": len(G.nodes()),
            "edges": len(G.edges()),
            "density": nx.density(G),
            "connected_components": nx.number_connected_components(G)
        }
        
        if len(G.nodes()) > 0:
            degrees = dict(G.degree())
            stats.update({
                "avg_degree": sum(degrees.values()) / len(degrees),
                "max_degree": max(degrees.values()) if degrees else 0,
                "min_degree": min(degrees.values()) if degrees else 0
            })
        
        return stats
    
    def _get_color_by_risk_type(self, risk_type: str) -> str:
        """Get color based on risk type"""
        color_map = {
            'Financial': '#ff6b6b',
            'Operational': '#4ecdc4', 
            'Strategic': '#45b7d1',
            'Credit': '#f9ca24',
            'Market': '#6c5ce7',
            'Unknown': '#95a5a6'
        }
        return color_map.get(risk_type, '#95a5a6')
    
    def _get_color_by_frequency(self, frequency: int) -> str:
        """Get color based on frequency"""
        if frequency >= 10:
            return '#e74c3c'  # Red for high frequency
        elif frequency >= 5:
            return '#f39c12'  # Orange for medium frequency
        else:
            return '#3498db'  # Blue for low frequency
    
    def _truncate_label(self, label: str, max_length: int = 20) -> str:
        """Truncate labels for better visualization"""
        if len(label) <= max_length:
            return label
        return label[:max_length-3] + "..."