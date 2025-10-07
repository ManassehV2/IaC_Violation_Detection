#!/usr/bin/env python3
"""
harvest_aws_terraform_repos_enhanced.py

AWS-Focused Enhanced Terraform Repository Harvester for 15k+ Quality AWS Repositories

This version focuses EXCLUSIVELY on AWS Terraform repositories for thesis work.

Improvements over the original:
1. AWS-only search strategies with comprehensive AWS service coverage
2. Broader time coverage (2018-2025)
3. AWS-specific quality filters and relevance validation
4. Smaller time intervals (bi-weekly) for better coverage
5. Enhanced AWS service pattern detection
6. Enhanced error handling and rate limit management

Target: 15,000+ high-quality AWS Terraform repositories
"""

import os
import time
import logging
from datetime import datetime
from argparse import ArgumentParser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import os

# === CONFIGURATION ===
# Set your GitHub token as an environment variable: export GITHUB_TOKEN=your_token_here
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')
if not GITHUB_TOKEN:
    print("Warning: No GitHub token found. Set GITHUB_TOKEN environment variable for higher rate limits.")
    print("You can still use the script but with lower rate limits (60 requests/hour vs 5000).")

API_URL = "https://api.github.com/search/repositories"
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}

# Add authorization header only if token is available
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

# AWS-FOCUSED search strategies for maximum AWS coverage
SEARCH_STRATEGIES = [
    # Strategy 1: Core AWS Terraform searches
    {
        "name": "aws_terraform_core",
        "queries": [
            "terraform aws language:HCL fork:false",
            "aws terraform language:HCL fork:false",
            "terraform amazon language:HCL fork:false",
            "terraform in:readme aws language:HCL fork:false",
        ]
    },
    # Strategy 2: AWS Compute Services
    {
        "name": "aws_compute",
        "queries": [
            "terraform ec2 language:HCL fork:false",
            "terraform lambda language:HCL fork:false",
            "terraform ecs language:HCL fork:false",
            "terraform eks language:HCL fork:false",
        ]
    },
    # Strategy 3: AWS Storage & Database Services
    {
        "name": "aws_storage_database",
        "queries": [
            "terraform s3 language:HCL fork:false",
            "terraform rds language:HCL fork:false",
            "terraform dynamodb language:HCL fork:false",
            "terraform ebs language:HCL fork:false",
        ]
    },
    # Strategy 4: AWS Networking Services
    {
        "name": "aws_networking",
        "queries": [
            "terraform vpc language:HCL fork:false",
            "terraform elb language:HCL fork:false",
            "terraform alb language:HCL fork:false",
            "terraform cloudfront language:HCL fork:false",
        ]
    },
    # Strategy 5: AWS Security & Management
    {
        "name": "aws_security_mgmt",
        "queries": [
            "terraform iam language:HCL fork:false",
            "terraform cloudwatch language:HCL fork:false",
            "terraform cloudtrail language:HCL fork:false",
            "terraform kms language:HCL fork:false",
        ]
    },
    # Strategy 6: AWS Topics and Infrastructure
    {
        "name": "aws_topics_infra",
        "queries": [
            "topic:aws language:HCL fork:false",
            "topic:aws-infrastructure language:HCL fork:false",
            "terraform in:description amazon web services language:HCL fork:false",
            "terraform aws infrastructure language:HCL fork:false",
        ]
    }
]

# Quality filters for better repositories
QUALITY_FILTERS = [
    "stars:>=2",          # At least 1 star
    "size:>=10",          # At least 10KB
    "pushed:>2018-01-01", # Updated since 2019
    "fork:false",
]

PER_PAGE = 100
MAX_PER_PERIOD = 2000  # Increased from 1000
MAX_PAGES = MAX_PER_PERIOD // PER_PAGE  # 20 pages

# === PARSE ARGS ===
parser = ArgumentParser(
    description="AWS-focused enhanced harvest of 15k+ quality AWS Terraform repositories"
)
parser.add_argument(
    "--since",
    type=str,
    default="2018-01-01",  # Extended time range
    help="Start date for repository creation (YYYY-MM-DD)",
)
parser.add_argument(
    "--until",
    type=str,
    default="2025-06-07",
    help="End date for repository creation (YYYY-MM-DD)",
)
parser.add_argument(
    "--outdir",
    type=str,
    default="repo_urls_aws_enhanced",
    help="Output directory for repository URLs",
)
parser.add_argument(
    "--target",
    type=int,
    default=15000,
    help="Target number of repositories to collect",
)

# === SETUP LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("aws_harvest_enhanced.log"),
        logging.StreamHandler()
    ]
)

# === SETUP SESSION ===
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
session.mount("https://", HTTPAdapter(max_retries=retries))

def check_rate_limit():
    """Check GitHub search API rate limit"""
    try:
        resp = session.get("https://api.github.com/rate_limit", headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            search_info = data.get("resources", {}).get("search", {})
            remaining = search_info.get("remaining", 0)
            reset_time = search_info.get("reset", 0)
            
            if remaining < 3:  # Less than 3 requests remaining
                wait_time = max(0, reset_time - int(time.time())) + 10
                logging.warning(f"Rate limit low ({remaining} remaining). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
                # Get fresh rate limit info after waiting
                try:
                    fresh_resp = session.get("https://api.github.com/rate_limit", headers=HEADERS)
                    if fresh_resp.status_code == 200:
                        fresh_data = fresh_resp.json()
                        fresh_search_info = fresh_data.get("resources", {}).get("search", {})
                        fresh_remaining = fresh_search_info.get("remaining", 30)
                        logging.info(f"Rate limit refreshed: {fresh_remaining} requests available")
                        return fresh_remaining
                except:
                    pass
                
                return 30  # Conservative fallback
                
            return remaining
    except Exception as e:
        logging.warning(f"Could not check rate limit: {e}")
        return 5  # Conservative estimate

def fetch_page(params):
    """Fetch a page from GitHub search API with rate limit handling"""
    remaining = check_rate_limit()
    
    # If we have very few requests left, be extra cautious
    if remaining < 2:
        logging.warning("Very few requests remaining. Waiting 60 seconds for rate limit reset...")
        time.sleep(60)
        remaining = check_rate_limit()
    
    try:
        resp = session.get(API_URL, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 403:
            # Rate limit hit, check if it's actually rate limit or other 403
            if "rate limit" in resp.text.lower() or "api rate limit" in resp.text.lower():
                logging.warning("Rate limit hit during request, waiting 60 seconds...")
                time.sleep(60)
                return fetch_page(params)  # Retry after waiting
            else:
                logging.error(f"Access forbidden (403) - not rate limit: {resp.text}")
                return None
        else:
            logging.error(f"API error {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        logging.error(f"Request failed: {e}")
        return None

def biweekly_ranges(start: datetime, end: datetime):
    """
    Yield bi-weekly (2-week) ranges for more granular searching
    """
    current = start
    while current < end:
        range_end = min(current + relativedelta(weeks=2), end)
        yield current, range_end
        current = range_end

def is_quality_repo(repo):
    """Check if repository meets quality criteria"""
    # Basic quality checks
    if repo.get("archived", False) or repo.get("disabled", False):
        return False
    
    # Must have some stars and reasonable size
    stars = repo.get("stargazers_count", 0)
    size = repo.get("size", 0)
    
    if stars < 1 or size < 10:
        return False
    
    # Check for meaningful description or topics
    description = repo.get("description", "") or ""
    topics = repo.get("topics", []) or []
    
    if not description.strip() and not topics:
        return False
    
    # Check for recent activity
    updated_at = repo.get("updated_at", "")
    if updated_at:
        try:
            updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            years_old = (datetime.now().astimezone() - updated_date).days / 365.25
            if years_old > 5:  # Not updated in 5 years
                return False
        except:
            pass
    
    return True

def is_aws_terraform_relevant(repo):
    """Check if repository is actually AWS Terraform-related"""
    # Comprehensive AWS keywords
    aws_keywords = {
        'aws', 'amazon', 'ec2', 'lambda', 's3', 'rds', 'vpc', 'iam', 
        'cloudwatch', 'cloudfront', 'dynamodb', 'eks', 'ecs', 'elb', 
        'alb', 'cloudtrail', 'kms', 'ebs', 'route53', 'apigateway',
        'cloudformation', 'elasticache', 'redshift', 'sqs', 'sns',
        'kinesis', 'glue', 'athena', 'emr', 'sagemaker'
    }
    
    terraform_keywords = {
        'terraform', 'hcl', 'infrastructure', 'iac', 'provisioning'
    }
    
    # Check name, description, and topics
    text_to_check = [
        repo.get("name", "").lower(),
        repo.get("description", "").lower() if repo.get("description") else "",
        " ".join(repo.get("topics", [])).lower()
    ]
    
    full_text = " ".join(text_to_check)
    
    # Must have both AWS and Terraform keywords
    has_aws = any(keyword in full_text for keyword in aws_keywords)
    has_terraform = any(keyword in full_text for keyword in terraform_keywords)
    
    return has_aws and has_terraform

def load_existing_repos(outdir: str):
    """Load existing repositories from output directory to avoid duplicates"""
    existing_repos = set()
    processed_files = set()
    
    if os.path.exists(outdir):
        for filename in os.listdir(outdir):
            if filename.endswith('.txt'):
                processed_files.add(filename)
                filepath = os.path.join(outdir, filename)
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            repo_url = line.strip()
                            if repo_url:
                                existing_repos.add(repo_url)
                except Exception as e:
                    logging.warning(f"Could not read {filename}: {e}")
    
    logging.info(f"Loaded {len(existing_repos)} existing repositories from {len(processed_files)} files")
    return existing_repos, processed_files

def collect_enhanced(start: datetime, end: datetime, outdir: str, target: int):
    """
    Enhanced AWS-focused collection using multiple strategies and time periods
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Load existing repositories to avoid duplicates
    all_repos, processed_files = load_existing_repos(outdir)
    total_collected = len(all_repos)
    
    logging.info(f"Starting AWS-focused enhanced collection from {start} to {end}")
    logging.info(f"Target: {target} repositories")
    logging.info(f"Already collected: {total_collected} repositories")
    logging.info(f"Using {len(SEARCH_STRATEGIES)} AWS-focused strategies with {sum(len(s['queries']) for s in SEARCH_STRATEGIES)} total queries")
    
    # Generate all time periods first
    time_periods = list(biweekly_ranges(start, end))
    logging.info(f"Generated {len(time_periods)} bi-weekly time periods")
    
    strategy_count = 0
    for strategy in SEARCH_STRATEGIES:
        strategy_count += 1
        logging.info(f"Strategy {strategy_count}/{len(SEARCH_STRATEGIES)}: {strategy['name']}")
        
        for query in strategy['queries']:
            logging.info(f"  Query: {query}")
            
            period_count = 0
            for period_start, period_end in time_periods:
                if total_collected >= target:
                    logging.info(f"Reached target of {target} repositories!")
                    break
                    
                period_count += 1
                
                # Check if this period was already processed for this strategy
                period_filename = f"{period_start.strftime('%Y-%m-%d')}_to_{period_end.strftime('%Y-%m-%d')}_{strategy['name']}.txt"
                if period_filename in processed_files:
                    logging.info(f"    Period {period_count}: Already processed (skipping)")
                    continue
                
                period_repos = set()
                
                # Build search parameters
                created_range = f"{period_start.strftime('%Y-%m-%d')}..{period_end.strftime('%Y-%m-%d')}"
                full_query = f"{query} created:{created_range}"
                
                # Add quality filters
                for quality_filter in QUALITY_FILTERS:
                    full_query += f" {quality_filter}"
                
                params = {
                    "q": full_query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": PER_PAGE,
                    "page": 1
                }
                
                # Fetch all pages for this period
                for page in range(1, MAX_PAGES + 1):
                    params["page"] = page
                    
                    data = fetch_page(params)
                    if not data or "items" not in data:
                        break
                    
                    repos = data["items"]
                    if not repos:
                        break
                    
                    for repo in repos:
                        if total_collected >= target:
                            break
                            
                        # Apply quality and relevance filters
                        if not is_quality_repo(repo):
                            continue
                            
                        if not is_aws_terraform_relevant(repo):
                            continue
                        
                        repo_url = repo["clone_url"]
                        if repo_url not in all_repos:
                            all_repos.add(repo_url)
                            period_repos.add(repo_url)
                            total_collected += 1
                    
                    # Small delay between pages
                    time.sleep(1)
                    
                    if total_collected >= target:
                        break
                
                # Save period results if any found
                if period_repos:
                    period_filename = f"{period_start.strftime('%Y-%m-%d')}_to_{period_end.strftime('%Y-%m-%d')}_{strategy['name']}.txt"
                    period_filepath = os.path.join(outdir, period_filename)
             
                    with open(period_filepath, "w") as f:
                        for url in sorted(period_repos):
                            f.write(f"{url}\n")
                    
                    logging.info(f"    Period {period_count}: {len(period_repos)} new AWS repos (Total: {total_collected})")
                
                if total_collected >= target:
                    break
            
            if total_collected >= target:
                break
        
        if total_collected >= target:
            break
    
    # Save consolidated results
    consolidated_file = os.path.join(outdir, "all_aws_terraform_repos.txt")
    with open(consolidated_file, "w") as f:
        for url in sorted(all_repos):
            f.write(f"{url}\n")
    
    logging.info(f"Collection complete!")
    logging.info(f"Total unique AWS Terraform repositories collected: {len(all_repos)}")
    logging.info(f"Results saved to: {outdir}")
    logging.info(f"Consolidated file: {consolidated_file}")
    
    return len(all_repos)

def main():
    """
    Parse args, convert dates, and run enhanced AWS harvest
    """
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.since, "%Y-%m-%d")
        end_date = datetime.strptime(args.until, "%Y-%m-%d")
    except ValueError as e:
        logging.error(f"Invalid date format: {e}")
        return
    
    logging.info("="*60)
    logging.info("AWS-FOCUSED ENHANCED TERRAFORM REPOSITORY HARVESTER")
    logging.info("="*60)
    logging.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logging.info(f"Output directory: {args.outdir}")
    logging.info(f"Target repositories: {args.target}")
    
    try:
        total_repos = collect_enhanced(start_date, end_date, args.outdir, args.target)
        logging.info(f"Successfully collected {total_repos} AWS Terraform repositories")
    except KeyboardInterrupt:
        logging.info("Collection interrupted by user")
    except Exception as e:
        logging.error(f"Collection failed: {e}")

if __name__ == "__main__":
    main()
