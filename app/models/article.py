from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    authors = Column(String(500))
    year = Column(Integer)
    journal = Column(String(255))
    objective = Column(Text)
    methodology = Column(Text)
    key_variables = Column(Text)
    risk_type = Column(String(255))
    level_of_analysis = Column(String(255))
    main_findings = Column(Text)
    implications = Column(Text)
    limitations = Column(Text) 