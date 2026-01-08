import re
import time
import logging
from urllib.parse import urlparse, urljoin
from collections import deque
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Email regex pattern
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

class OptimizedWebScraper:
    def __init__(self, timeout=8, delay=0.3, max_pages=15, max_workers=5):
        self.timeout = timeout
        self.delay = delay
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Cache for visited URLs and emails
        self.url_cache = {}
        self.domain_cache = {}
        self.email_cache = {}
        self.lock = threading.Lock()
        
        # Priority pages to check first (speeds up email discovery)
        self.priority_paths = [
            '/',
            '/contact',
            '/about',
            '/team',
            '/contact-us',
            '/about-us',
            '/help',
            '/support',
            '/careers'
        ]
    
    @lru_cache(maxsize=1000)
    def is_valid_url(self, url):
        """Validate URL format with caching"""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False
    
    @lru_cache(maxsize=1000)
    def normalize_url(self, url):
        """Normalize URL to base domain with caching"""
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except:
            return url
    
    @lru_cache(maxsize=1000)
    def get_domain(self, url):
        """Extract domain from URL with caching"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return None
    
    def extract_emails_from_text(self, text):
        """Extract emails from text using regex"""
        if not text:
            return set()
        emails = set()
        found_emails = re.findall(EMAIL_PATTERN, text)
        emails.update(found_emails)
        return emails
    
    def extract_emails_from_links(self, soup):
        """Extract emails from mailto links"""
        emails = set()
        try:
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip()
                if href.startswith('mailto:'):
                    email = href.replace('mailto:', '').split('?')[0].strip()
                    if re.match(EMAIL_PATTERN, email):
                        emails.add(email)
        except Exception as e:
            logger.debug(f"Error extracting emails from links: {str(e)}")
        return emails
    
    def extract_links_from_page(self, soup, base_url):
        """Extract only priority links for faster processing"""
        links = set()
        base_domain = self.normalize_url(base_url)
        base_parsed = urlparse(base_url)
        base_path = base_parsed.path.rstrip('/')
        
        try:
            # First, add priority paths
            for priority_path in self.priority_paths:
                priority_url = f"{base_domain}{priority_path}".rstrip('/')
                if priority_url:
                    links.add(priority_url)
            
            # Then extract links from page
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip()
                
                if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:', 'ftp:')):
                    continue
                
                try:
                    absolute_url = urljoin(base_url, href)
                except:
                    continue
                
                # Only include links from the same domain
                if self.normalize_url(absolute_url) == base_domain:
                    parsed = urlparse(absolute_url)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
                    
                    if clean_url:
                        links.add(clean_url)
        except Exception as e:
            logger.debug(f"Error extracting links: {str(e)}")
        
        return links
    
    def scrape_single_page(self, url, base_domain=None):
        """Scrape a single page for emails and links with caching"""
        
        # Check cache first
        with self.lock:
            if url in self.url_cache:
                return self.url_cache[url]
        
        emails = set()
        links = set()
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout, verify=True)
            
            if response.status_code != 200:
                logger.debug(f"Status code {response.status_code} for {url}")
                result = (emails, links)
                with self.lock:
                    self.url_cache[url] = result
                return result
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'noscript']):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Extract emails from text
            emails.update(self.extract_emails_from_text(text))
            
            # Extract emails from mailto links
            emails.update(self.extract_emails_from_links(soup))
            
            # Extract internal links
            links.update(self.extract_links_from_page(soup, url))
            
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout while accessing {url}")
        except requests.exceptions.ConnectionError:
            logger.debug(f"Connection error while accessing {url}")
        except Exception as e:
            logger.debug(f"Error scraping {url}: {str(e)}")
        
        result = (emails, links)
        
        # Cache the result
        with self.lock:
            self.url_cache[url] = result
        
        return result
    
    def scrape_website_optimized(self, base_url):
        """
        Scrape entire website with optimization:
        - Uses BFS with priority paths first (faster email discovery)
        - Limits depth and breadth
        - Uses caching to avoid redundant requests
        
        Args:
            base_url: Starting URL
            
        Returns:
            Dictionary with emails and crawl statistics
        """
        logger.info(f"Starting optimized scrape of {base_url}")
        
        # Validate and normalize URL
        if not self.is_valid_url(base_url):
            if not base_url.startswith(('http://', 'https://')):
                base_url = 'https://' + base_url
        
        if not self.is_valid_url(base_url):
            logger.error(f"Invalid URL: {base_url}")
            return {
                'base_url': base_url,
                'emails': [],
                'pages_crawled': 0,
                'error': 'Invalid URL format'
            }
        
        base_url = base_url.rstrip('/')
        base_domain = self.normalize_url(base_url)
        
        # Initialize tracking
        all_emails = set()
        visited_urls = set()
        urls_to_visit = deque([base_url])
        pages_crawled = 0
        start_time = time.time()
        
        # Scrape pages
        while urls_to_visit and pages_crawled < self.max_pages:
            current_url = urls_to_visit.popleft()
            
            # Skip if already visited
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            logger.info(f"[{pages_crawled + 1}/{self.max_pages}] Crawling: {current_url}")
            
            # Scrape the page
            emails, links = self.scrape_single_page(current_url, base_domain)
            all_emails.update(emails)
            pages_crawled += 1
            
            # Early exit if many emails found
            if len(all_emails) > 50:
                logger.info(f"Found {len(all_emails)} emails. Stopping early to save time.")
                break
            
            # Add new links to queue (limit by max_pages)
            for link in sorted(links):
                if link not in visited_urls and len(visited_urls) < self.max_pages:
                    urls_to_visit.append(link)
            
            # Adaptive delay based on batch size
            time.sleep(self.delay)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed {base_url} - Found {len(all_emails)} emails in {pages_crawled} pages ({elapsed_time:.2f}s)")
        
        return {
            'base_url': base_url,
            'emails': sorted(list(all_emails)),
            'pages_crawled': pages_crawled,
            'email_count': len(all_emails),
            'time_taken': elapsed_time,
            'error': None
        }
    
    def scrape_multiple_urls_parallel(self, urls):
        """
        Scrape multiple URLs in parallel using ThreadPoolExecutor
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of results for each URL
        """
        results = []
        
        logger.info(f"Starting parallel scraping of {len(urls)} URLs with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.scrape_website_optimized, url): url 
                for url in urls if url and url.strip() and url.lower() != 'nan'
            }
            
            # Collect results as they complete
            completed = 0
            total = len(future_to_url)
            
            for future in as_completed(future_to_url):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    logger.info(f"Completed {completed}/{total} URLs")
                except Exception as e:
                    url = future_to_url[future]
                    logger.error(f"Error scraping {url}: {str(e)}")
                    results.append({
                        'base_url': url,
                        'emails': [],
                        'pages_crawled': 0,
                        'time_taken': 0,
                        'error': str(e)
                    })
        
        return results
    
    def save_results_to_excel(self, results, filename='scraped_emails.xlsx'):
        """
        Save scraping results to Excel file with formatting
        
        Args:
            results: List of results from scrape_multiple_urls_parallel
            filename: Output Excel filename
        """
        data = []
        
        for result in results:
            if result['error']:
                data.append({
                    'Company URL': result['base_url'],
                    'Email': f"Error: {result['error']}",
                    'Status': 'Failed',
                    'Pages Crawled': 0,
                    'Time (s)': result.get('time_taken', 0)
                })
            elif result['email_count'] == 0:
                data.append({
                    'Company URL': result['base_url'],
                    'Email': 'No emails found',
                    'Status': 'No Results',
                    'Pages Crawled': result['pages_crawled'],
                    'Time (s)': result.get('time_taken', 0)
                })
            else:
                for email in result['emails']:
                    data.append({
                        'Company URL': result['base_url'],
                        'Email': email,
                        'Status': 'Success',
                        'Pages Crawled': result['pages_crawled'],
                        'Time (s)': round(result.get('time_taken', 0), 2)
                    })
        
        df = pd.DataFrame(data)
        
        # Write to Excel with formatting
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            
            # Format columns
            worksheet = writer.sheets['Results']
            worksheet.column_dimensions['A'].width = 40
            worksheet.column_dimensions['B'].width = 35
            worksheet.column_dimensions['C'].width = 15
            worksheet.column_dimensions['D'].width = 15
            worksheet.column_dimensions['E'].width = 12
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    def print_results(self, results):
        """Print results to console"""
        print("\n" + "="*100)
        print("SCRAPING RESULTS - OPTIMIZED WITH PARALLEL PROCESSING")
        print("="*100)
        
        total_emails = 0
        total_time = 0
        
        for idx, result in enumerate(results, 1):
            print(f"\n[{idx}] URL: {result['base_url']}")
            print(f"    Pages Crawled: {result['pages_crawled']}")
            print(f"    Time Taken: {result.get('time_taken', 0):.2f}s")
            
            if result['error']:
                print(f"    Error: {result['error']}")
            elif result['email_count'] == 0:
                print("    No emails found")
            else:
                print(f"    Emails Found ({result['email_count']}):")
                for email in result['emails']:
                    print(f"      - {email}")
                total_emails += result['email_count']
            
            total_time += result.get('time_taken', 0)
        
        print("\n" + "="*100)
        print(f"SUMMARY: Total {len(results)} URLs | {total_emails} Emails Found | {total_time:.2f}s Total")
        print("="*100 + "\n")


# Main Function - Simple to Use
def scrape_from_excel(input_file, output_file='scraped_emails.xlsx', max_workers=5, max_pages=15):
    """
    Simple function to scrape URLs from Excel file
    
    Args:
        input_file: Excel file with 'Urls' column
        output_file: Output Excel file with results
        max_workers: Number of parallel workers (default 5)
        max_pages: Max pages to crawl per URL (default 15)
    """
    try:
        # Read URLs from Excel
        print(f"Reading URLs from {input_file}...")
        df = pd.read_excel(input_file)
        
        if 'Urls' not in df.columns:
            print("Error: Excel file must have 'Urls' column")
            return
        
        urls = df['Urls'].dropna().astype(str).tolist()
        print(f"Found {len(urls)} URLs to scrape\n")
        
        # Initialize scraper with optimizations
        scraper = OptimizedWebScraper(
            timeout=8,
            delay=0.3,
            max_pages=max_pages,
            max_workers=max_workers
        )
        
        # Scrape all URLs in parallel
        start_time = time.time()
        results = scraper.scrape_multiple_urls_parallel(urls)
        total_time = time.time() - start_time
        
        # Print results
        scraper.print_results(results)
        
        # Save to Excel
        output_path = scraper.save_results_to_excel(results, output_file)
        
        # Print summary stats
        total_emails = sum(r['email_count'] for r in results if not r['error'])
        successful_urls = len([r for r in results if not r['error'] and r['email_count'] > 0])
        
        print(f"\n✅ FINAL STATISTICS:")
        print(f"   Total URLs: {len(urls)}")
        print(f"   Successful: {successful_urls}")
        print(f"   Total Emails Found: {total_emails}")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Avg Time per URL: {total_time/len(urls):.2f} seconds")
        print(f"   Output File: {output_path}\n")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")


# Usage Example
if __name__ == "__main__":
    # Simple one-line usage
    scrape_from_excel(
        input_file='tech_urls.xlsx',
        output_file='scraped_emails2.xlsx',
        max_workers=5,      # Use 5 parallel workers
        max_pages=25        # Crawl max 15 pages per URL
    )