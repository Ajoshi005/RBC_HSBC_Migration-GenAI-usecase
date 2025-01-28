# this module creates and stores a DB of crawled pages from a website.
# this DB will be used to create a knowledge base for the chatbot.
# the DB will be converted to a index in pinecone and will be used to retrieve the most relevant documents.
from xml.etree import ElementTree
import asyncio, requests
from typing import List, Optional
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator
from pydantic import BaseModel, Field, HttpUrl
from typing import List

class CrawledData(BaseModel):
    '''
    CrawledData class to store the crawled data. Add fields needed
    to identify and summarise content to be added as metadata
    '''
    url: HttpUrl = Field(..., description="Full URL of the webpage")
   
    # Content fields
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Main content of the webpage")
    summary: str = Field(..., description="AI-generated summary of content")

    # Timestamps and tracking
    crawl_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Date and time of crawl in UTC"
    )
    last_modified: Optional[datetime] = Field(
        None, 
        description="Last modified date from webpage headers"
    )
# Create an instance
page = CrawledData(
    url="https://www.langchain.com/about",
    title="Page Title",
    content="Main content here",
    summary="Brief summary"
)



def ai_docs_urls():
    """
    Fetches all URLs from the given sitemap.
    Uses the sitemap to get these URLs to scrape.
    
    Returns:
        List[str]: List of URLs
    """            
    sitemap_url = "https://www.langchain.com/sitemap.xml"
    # "https://www.rbcbank.com/sitemap.xml" # provide sitemap url
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        # The namespace is usually defined in the root element
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []
    
# print(ai_docs_urls())


async def crawl_sequential(urls: List[str]):
    """
    gets a list of URLs and scrapes the content of each URL sequentially.
    Uses the AsyncWebCrawler to scrape the content of each URL.
    
    Returns:
        Dict[str, str]: Dictionary of URL and its content
    """  
    print("\n=== Sequential Crawling with Session Reuse ===")

    browser_config = BrowserConfig(
        headless=True,
        # For better performance in Docker or low-memory environments:
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )

    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator()
    )

    # Create the crawler (opens the browser)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        session_id = "session1"  # Reuse the same session across all URLs
        url_content = {}
        for url in urls:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id
            )
            if result.success:
                url_content[url] = result.markdown_v2.raw_markdown
                print(f"Successfully crawled: {url}")
                # E.g. check markdown length
                print(f"Markdown length: {len(result.markdown_v2.raw_markdown)}")
            else:
                print(f"Failed: {url} - Error: {result.error_message}")
                url_content[url] = ""
    finally:
        # After all URLs are done, close the crawler (and the browser)
        await crawler.close()
        return url_content