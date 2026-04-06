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
from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask App Setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['JSON_SORT_KEYS'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        self.url_cache = {}
        self.lock = threading.Lock()
        self.priority_paths = ['/', '/contact', '/about', '/team', '/contact-us', '/about-us', '/help', '/support', '/careers']
    
    @lru_cache(maxsize=1000)
    def is_valid_url(self, url):
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False
    
    @lru_cache(maxsize=1000)
    def normalize_url(self, url):
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except:
            return url
    
    def extract_emails_from_text(self, text):
        if not text:
            return set()
        emails = set()
        found_emails = re.findall(EMAIL_PATTERN, text)
        emails.update(found_emails)
        return emails
    
    def extract_emails_from_links(self, soup):
        emails = set()
        try:
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip()
                if href.startswith('mailto:'):
                    email = href.replace('mailto:', '').split('?')[0].strip()
                    if re.match(EMAIL_PATTERN, email):
                        emails.add(email)
        except:
            pass
        return emails
    
    def extract_links_from_page(self, soup, base_url):
        links = set()
        base_domain = self.normalize_url(base_url)
        
        try:
            for priority_path in self.priority_paths:
                priority_url = f"{base_domain}{priority_path}".rstrip('/')
                if priority_url:
                    links.add(priority_url)
            
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip()
                
                if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:', 'ftp:')):
                    continue
                
                try:
                    absolute_url = urljoin(base_url, href)
                except:
                    continue
                
                if self.normalize_url(absolute_url) == base_domain:
                    parsed = urlparse(absolute_url)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
                    if clean_url:
                        links.add(clean_url)
        except:
            pass
        
        return links
    
    def scrape_single_page(self, url):
        with self.lock:
            if url in self.url_cache:
                return self.url_cache[url]
        
        emails = set()
        links = set()
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout, verify=True)
            
            if response.status_code != 200:
                result = (emails, links)
                with self.lock:
                    self.url_cache[url] = result
                return result
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(['script', 'style', 'noscript']):
                script.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            emails.update(self.extract_emails_from_text(text))
            emails.update(self.extract_emails_from_links(soup))
            links.update(self.extract_links_from_page(soup, url))
            
        except:
            pass
        
        result = (emails, links)
        with self.lock:
            self.url_cache[url] = result
        
        return result
    
    def scrape_website_optimized(self, base_url):
        if not self.is_valid_url(base_url):
            if not base_url.startswith(('http://', 'https://')):
                base_url = 'https://' + base_url
        
        if not self.is_valid_url(base_url):
            return {
                'base_url': base_url,
                'emails': [],
                'pages_crawled': 0,
                'error': 'Invalid URL format'
            }
        
        base_url = base_url.rstrip('/')
        all_emails = set()
        visited_urls = set()
        urls_to_visit = deque([base_url])
        pages_crawled = 0
        start_time = time.time()
        
        while urls_to_visit and pages_crawled < self.max_pages:
            current_url = urls_to_visit.popleft()
            
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            emails, links = self.scrape_single_page(current_url)
            all_emails.update(emails)
            pages_crawled += 1
            
            if len(all_emails) > 50:
                break
            
            for link in sorted(links):
                if link not in visited_urls and len(visited_urls) < self.max_pages:
                    urls_to_visit.append(link)
            
            time.sleep(self.delay)
        
        elapsed_time = time.time() - start_time
        
        return {
            'base_url': base_url,
            'emails': sorted(list(all_emails)),
            'pages_crawled': pages_crawled,
            'email_count': len(all_emails),
            'time_taken': elapsed_time,
            'error': None
        }
    
    def scrape_multiple_urls_parallel(self, urls):
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.scrape_website_optimized, url): url 
                for url in urls if url and url.strip() and url.lower() != 'nan'
            }
            
            for future in as_completed(future_to_url):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    url = future_to_url[future]
                    results.append({
                        'base_url': url,
                        'emails': [],
                        'pages_crawled': 0,
                        'time_taken': 0,
                        'error': str(e)
                    })
        
        return results
    
    def save_results_to_excel(self, results, filename='scraped_emails.xlsx'):
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
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            worksheet = writer.sheets['Results']
            worksheet.column_dimensions['A'].width = 40
            worksheet.column_dimensions['B'].width = 35
            worksheet.column_dimensions['C'].width = 15
            worksheet.column_dimensions['D'].width = 15
            worksheet.column_dimensions['E'].width = 12
        
        output.seek(0)
        return output

# Global scraper instance
scraper = OptimizedWebScraper(timeout=8, delay=0.3, max_pages=15, max_workers=5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process-excel', methods=['POST'])
def process_excel():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            return jsonify({'success': False, 'error': 'Invalid file format'}), 400
        
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading file: {str(e)}'}), 400
        
        if 'Urls' not in df.columns:
            return jsonify({'success': False, 'error': 'Excel file must have "Urls" column'}), 400
        
        urls = df['Urls'].dropna().astype(str).tolist()
        
        if not urls:
            return jsonify({'success': False, 'error': 'No URLs found'}), 400
        
        results = scraper.scrape_multiple_urls_parallel(urls)
        
        total_emails = sum(r['email_count'] for r in results if not r['error'])
        successful_urls = len([r for r in results if not r['error'] and r['email_count'] > 0])
        
        return jsonify({
            'success': True,
            'results': results,
            'total_urls': len(urls),
            'successful': successful_urls,
            'total_emails': total_emails
        }), 200
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'success': False, 'error': 'Server error'}), 500

@app.route('/api/download', methods=['POST'])
def download():
    try:
        data = request.get_json()
        results = data.get('results', [])
        
        if not results:
            return jsonify({'success': False, 'error': 'No data to download'}), 400
        
        excel_file = scraper.save_results_to_excel(results)
        
        return send_file(
            excel_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='scraped_emails.xlsx'
        )
    
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'success': False, 'error': 'Error creating download file'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)