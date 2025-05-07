from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from typing import List
import os
import json

class GoogleSheetsHandler:
    def __init__(self):
        try:
            # Use service account credentials
            SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
            
            # Print the current working directory
            print(f"Current working directory: {os.getcwd()}")
            
            # Verify credentials file exists
            if not os.path.exists('credentials.json'):
                raise FileNotFoundError("credentials.json not found in current directory")
            
            credentials = service_account.Credentials.from_service_account_file(
                'credentials.json', scopes=SCOPES)
            
            self.service = build('sheets', 'v4', credentials=credentials)
            self.spreadsheet_id = os.getenv('SPREADSHEET_ID')
            
            if not self.spreadsheet_id:
                raise ValueError("SPREADSHEET_ID not found in environment variables")
                
            print(f"Successfully initialized Google Sheets handler with spreadsheet ID: {self.spreadsheet_id}")
            
        except Exception as e:
            print(f"Error initializing Google Sheets handler: {str(e)}")
            raise

    def initialize_sheet(self):
        try:
            print("Starting sheet initialization...")
            sheet_metadata = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id).execute()
            
            print("Successfully got sheet metadata")
            print(f"Available sheets: {json.dumps(sheet_metadata.get('sheets', []), indent=2)}")
            
            # Check if "Articles" sheet exists
            sheet_exists = False
            for sheet in sheet_metadata.get('sheets', ''):
                if sheet.get('properties', {}).get('title', '') == 'Articles':
                    sheet_exists = True
                    break
            
            if not sheet_exists:
                print("Creating new 'Articles' sheet...")
                request = {
                    'addSheet': {
                        'properties': {
                            'title': 'Articles'
                        }
                    }
                }
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body={'requests': [request]}
                ).execute()
                print("'Articles' sheet created successfully")

            # Define headers
            headers = [
                'Title', 'Authors', 'Year', 'Journal', 'Objective', 'Methodology',
                'Key Variables', 'Risk Type', 'Level of Analysis', 'Main Findings',
                'Implications', 'Limitations'
            ]
            
            print("Updating headers...")
            body = {
                'values': [headers]
            }
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range='Articles!A1',
                valueInputOption='RAW',
                body=body
            ).execute()
            print("Headers updated successfully")
            
        except Exception as e:
            print(f"Error in initialize_sheet: {str(e)}")
            raise

    def append_articles(self, articles: List[dict]):
        try:
            print(f"Starting to append {len(articles)} articles...")
            rows = []
            for article in articles:
                row = [
                    article.get('title', ''),
                    ', '.join(article.get('authors', [])),
                    str(article.get('year', '')),
                    article.get('journal', ''),
                    article.get('objective', ''),
                    article.get('methodology', ''),
                    article.get('key_variables', ''),
                    article.get('risk_type', ''),
                    article.get('level_of_analysis', ''),
                    article.get('main_findings', ''),
                    article.get('implications', ''),
                    article.get('limitations', '')
                ]
                rows.append(row)

            body = {
                'values': rows
            }
            
            print("Sending append request to Google Sheets...")
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range='Articles!A2',
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            print(f"Append result: {json.dumps(result, indent=2)}")
            
        except Exception as e:
            print(f"Error in append_articles: {str(e)}")
            raise 