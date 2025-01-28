# this module creates and stores a DB of crawled pages from a website.
# this DB will be used to create a knowledge base for the chatbot.
# the DB will be converted to a index in pinecone and will be used to retrieve the most relevant documents.
from xml.etree import ElementTree
import asyncio, requests
from typing import List, Optional
from datetime import datetime, timezone
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator
from pydantic import BaseModel, Field, HttpUrl


class CrawledData(BaseModel):
    '''
    CrawledData class to store the crawled data. Add fields needed
    to identify and summarise content to be added as metadata
    '''
    url: HttpUrl = Field(..., description="Full URL of the webpage")
   
    # Content fields
    # title: str = Field(..., description="Page title")
    content: str = Field(..., description="Main content of the webpage")
    # summary: str = Field(..., description="AI-generated summary of content")

    # Timestamps and tracking
    crawl_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),  # Default to current time
        description="Date and time of crawl in UTC"
    )
    last_modified: Optional[datetime] = Field(
        None,
        description="Last modified date from webpage headers"
    )
# Create an instance
page = CrawledData(
    url="https://www.langchain.com/about",
    # title="Page Title",
    content="Main content here",
    # summary="Brief summary"
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
        urls = [loc.text.strip() for loc in root.findall('.//ns:loc', namespace)]

        
        return urls
    except requests.RequestException as e:  # More specific exception
        print(f"Error fetching sitemap: {e}")
        return []
    
print(ai_docs_urls())


async def crawl_sequential(urls: List[str]) -> List[CrawledData]:
    """
    gets a list of URLs and scrapes the content of each URL sequentially.
    Uses the AsyncWebCrawler to scrape the content of each URL.
    
    Returns:
        List[CrawledData]: List of CrawledData objects containing URL and content
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
        crawled_data = []
        for url in urls:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id
            )
            if result.success:
                # Create CrawledData object for successful crawls
                page_data = CrawledData(
                    url=url,
                    content=result.markdown_v2.raw_markdown,
                    # last_modified will default to None as per your model
                )
                crawled_data.append(page_data)
                print(f"Successfully crawled: {url}")
                print(f"Markdown length: {len(result.markdown_v2.raw_markdown)}")
            else:
                print(f"Failed: {url} - Error: {result.error_message}")
                # For failed crawls, you might want to still create an object with empty content
                # or simply skip it based on your needs
                page_data = CrawledData(
                    url=url,
                    content="",
                )
                crawled_data.append(page_data)
    finally:
        # After all URLs are done, close the crawler (and the browser)
        await crawler.close()
        return crawled_data
    

async def main():
    try:
        urls = ai_docs_urls()
        #['https://www.langchain.com/langchain', 'https://www.langchain.com/about']
        
        if not urls:
            print("No URLs found to crawl")
            return
        crawled_pages = await crawl_sequential(urls)
        # Access the data
        for page in crawled_pages:
            print(f"URL: {page.url}")
            print(f"Content length: {len(page.content)}")
            print(f"Crawl date: {page.crawl_date}")
    except Exception as e:
        print(f"Error in main: {e}")

# Run the main function
asyncio.run(main())