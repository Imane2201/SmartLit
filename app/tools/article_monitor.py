import asyncio
import schedule
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from .crossref import CrossRefSearchTool
from .article_analyzer import ArticleAnalyzer
from .vector_store import VectorStoreService
from .sheets_handler import GoogleSheetsHandler


class ArticleMonitor:
    def __init__(self):
        """Initialize the article monitor with all necessary services"""
        self.crossref_tool = CrossRefSearchTool()
        self.article_analyzer = ArticleAnalyzer()
        self.vector_store = VectorStoreService()
        self.sheets_handler = GoogleSheetsHandler()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Track processed articles to avoid duplicates
        self.processed_titles = set()
        
        # Default search topics
        self.default_topics = [
            "financial risk management",
            "operational risk",
            "credit risk assessment",
            "market risk analysis",
            "enterprise risk management"
        ]
        
        # Monitoring configuration
        self.config = {
            "max_articles_per_topic": 5,
            "min_days_since_publication": 1,
            "max_days_since_publication": 30,
            "enabled": True
        }
    
    def load_processed_titles(self, filepath: str = "processed_articles.json") -> None:
        """Load previously processed article titles from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.processed_titles = set(data.get('processed_titles', []))
                self.logger.info(f"Loaded {len(self.processed_titles)} processed article titles")
        except FileNotFoundError:
            self.logger.info("No previous processed articles file found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading processed titles: {str(e)}")
    
    def save_processed_titles(self, filepath: str = "processed_articles.json") -> None:
        """Save processed article titles to file"""
        try:
            data = {
                "processed_titles": list(self.processed_titles),
                "last_updated": datetime.now().isoformat()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.processed_titles)} processed article titles")
        except Exception as e:
            self.logger.error(f"Error saving processed titles: {str(e)}")
    
    def is_article_recent(self, article: Dict[str, Any]) -> bool:
        """Check if article is within the specified date range"""
        year = article.get('year')
        if not year:
            return False
        
        # Simple check based on year (could be enhanced with full date parsing)
        current_year = datetime.now().year
        min_year = current_year - (self.config["max_days_since_publication"] // 365)
        
        return year >= min_year
    
    async def search_and_analyze_new_articles(self, topic: str) -> List[Dict[str, Any]]:
        """
        Search for new articles on a topic and analyze them
        
        Args:
            topic: Search topic
            
        Returns:
            List of newly processed articles
        """
        if not self.config["enabled"]:
            return []
        
        try:
            self.logger.info(f"Searching for new articles on topic: {topic}")
            
            # Search for articles
            articles = self.crossref_tool._run(topic)
            
            # Filter for new and recent articles
            new_articles = []
            for article in articles[:self.config["max_articles_per_topic"]]:
                title = article.get('title', '')
                
                # Skip if already processed
                if title in self.processed_titles:
                    continue
                
                # Skip if not recent enough
                if not self.is_article_recent(article):
                    continue
                
                # Skip if no abstract
                if not article.get('abstract'):
                    continue
                
                new_articles.append(article)
            
            if not new_articles:
                self.logger.info(f"No new articles found for topic: {topic}")
                return []
            
            self.logger.info(f"Found {len(new_articles)} new articles for topic: {topic}")
            
            # Analyze the new articles
            analyzed_articles = []
            for article in new_articles:
                try:
                    # Analyze article
                    analysis, token_usage = await self.article_analyzer.analyze(article["abstract"])
                    
                    # Combine metadata with analysis
                    full_article = {**article, **analysis}
                    full_article["processed_date"] = datetime.now().isoformat()
                    full_article["monitoring_topic"] = topic
                    
                    analyzed_articles.append(full_article)
                    
                    # Mark as processed
                    self.processed_titles.add(article.get('title', ''))
                    
                    self.logger.info(f"Analyzed article: {article.get('title', 'Unknown')[:50]}...")
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing article '{article.get('title', 'Unknown')}': {str(e)}")
                    continue
            
            return analyzed_articles
            
        except Exception as e:
            self.logger.error(f"Error searching for articles on topic '{topic}': {str(e)}")
            return []
    
    async def process_all_topics(self) -> Dict[str, Any]:
        """
        Process all configured topics and return summary
        
        Returns:
            Summary of processing results
        """
        if not self.config["enabled"]:
            self.logger.info("Article monitoring is disabled")
            return {"status": "disabled", "articles_processed": 0}
        
        start_time = datetime.now()
        all_articles = []
        topic_results = {}
        
        self.logger.info("Starting article monitoring cycle")
        
        # Load previously processed titles
        self.load_processed_titles()
        
        # Process each topic
        for topic in self.default_topics:
            try:
                articles = await self.search_and_analyze_new_articles(topic)
                all_articles.extend(articles)
                topic_results[topic] = len(articles)
                
                # Small delay between topics to be respectful to APIs
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error processing topic '{topic}': {str(e)}")
                topic_results[topic] = 0
        
        # Store results if any articles were found
        if all_articles:
            try:
                # Add to Google Sheets
                self.sheets_handler.append_articles(all_articles)
                self.logger.info(f"Added {len(all_articles)} articles to Google Sheets")
                
                # Add to vector store
                vector_stats = self.vector_store.add_articles(all_articles)
                self.logger.info(f"Added {vector_stats['total_chunks']} chunks to vector store")
                
            except Exception as e:
                self.logger.error(f"Error storing articles: {str(e)}")
        
        # Save processed titles
        self.save_processed_titles()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "status": "completed",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_articles_processed": len(all_articles),
            "topics_processed": len(self.default_topics),
            "topic_results": topic_results,
            "articles": all_articles
        }
        
        self.logger.info(f"Monitoring cycle completed: {len(all_articles)} new articles processed in {duration:.1f} seconds")
        
        return summary
    
    def add_topic(self, topic: str) -> None:
        """Add a new topic to monitor"""
        if topic not in self.default_topics:
            self.default_topics.append(topic)
            self.logger.info(f"Added new monitoring topic: {topic}")
    
    def remove_topic(self, topic: str) -> bool:
        """Remove a topic from monitoring"""
        if topic in self.default_topics:
            self.default_topics.remove(topic)
            self.logger.info(f"Removed monitoring topic: {topic}")
            return True
        return False
    
    def update_config(self, **kwargs) -> None:
        """Update monitoring configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                self.logger.info(f"Updated config {key} = {value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "enabled": self.config["enabled"],
            "topics": self.default_topics,
            "config": self.config,
            "processed_articles_count": len(self.processed_titles)
        }
    
    def schedule_monitoring(self, interval_hours: int = 24) -> None:
        """
        Schedule regular monitoring
        
        Args:
            interval_hours: Hours between monitoring cycles
        """
        def run_monitoring():
            """Wrapper to run async monitoring in sync context"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.process_all_topics())
                self.logger.info(f"Scheduled monitoring completed: {result['total_articles_processed']} articles")
            except Exception as e:
                self.logger.error(f"Error in scheduled monitoring: {str(e)}")
            finally:
                loop.close()
        
        # Schedule the monitoring
        schedule.every(interval_hours).hours.do(run_monitoring)
        
        self.logger.info(f"Scheduled monitoring every {interval_hours} hours")
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour


# Utility function to run monitoring manually
async def run_manual_monitoring(topics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run monitoring manually for specific topics
    
    Args:
        topics: Optional list of topics to monitor (uses defaults if None)
        
    Returns:
        Summary of monitoring results
    """
    monitor = ArticleMonitor()
    
    if topics:
        monitor.default_topics = topics
    
    return await monitor.process_all_topics()