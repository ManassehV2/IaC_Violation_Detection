#!/usr/bin/env python3
"""
Script to clean AWS credentials and other sensitive data from dataset CSV files.
This replaces real AWS access keys and secrets with obviously fake placeholder values.
"""

import re
import csv
import os
import sys

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

def clean_aws_credentials(text):
    """Replace AWS access keys and secrets with placeholder values"""
    if not text:
        return text
    
    # Replace AWS Access Key IDs (AKIA followed by 16 alphanumeric characters)
    text = re.sub(r'AKIA[0-9A-Z]{16}', 'AKIAXXXXXXXXXXXXXXXX', text)
    
    # Replace potential AWS Secret Access Keys (40+ character alphanumeric strings)
    # Look for patterns that appear after secret key indicators
    text = re.sub(r'(secret_key\s*=\s*["\'])[A-Za-z0-9+/=]{30,}(["\'])', r'\1XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\2', text)
    text = re.sub(r'(aws_secret_access_key\s*["\']?\s*[=:]\s*["\']?)[A-Za-z0-9+/=]{30,}(["\']?)', r'\1XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\2', text)
    text = re.sub(r'(AWS_SECRET_ACCESS_KEY\s*[=]\s*)[A-Za-z0-9+/=]{30,}', r'\1XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', text)
    
    # Replace standalone long alphanumeric strings that look like secrets (be more conservative)
    # Only replace if they're in quotes or follow equals signs
    text = re.sub(r'(["\'][A-Za-z0-9+/=]{40}["\'])', '\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\"', text)
    
    return text

def clean_csv_file(file_path):
    """Clean a CSV file by replacing AWS credentials with placeholders"""
    print(f"Cleaning {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Read the original file
    rows = []
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Clean each cell in the row
            cleaned_row = [clean_aws_credentials(cell) for cell in row]
            rows.append(cleaned_row)
    
    # Write back the cleaned data
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"✓ Cleaned {file_path}")

def main():
    dataset_dir = "/Users/minase/Desktop/EDISS_Resources/Thesis/Thesis_Repo/IaC_Violation_Detection/dataset"
    
    # Clean all CSV files in the dataset directory
    csv_files = [
        "aws_eval.csv",
        "aws_gold.csv", 
        "aws_train.csv"
    ]
    
    for csv_file in csv_files:
        file_path = os.path.join(dataset_dir, csv_file)
        clean_csv_file(file_path)
    
    print("\n✓ All dataset files have been cleaned!")
    print("AWS credentials have been replaced with placeholder values.")

if __name__ == "__main__":
    main()