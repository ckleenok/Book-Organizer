import math
import re
from datetime import date, datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from pyvis.network import Network
import streamlit.components.v1 as components
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = None


APP_TITLE = "Book Organizer – Mind Map"


def initialize_session_state() -> None:
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "mindmap_html" not in st.session_state:
        st.session_state.mindmap_html = None
    if "book_title" not in st.session_state:
        st.session_state.book_title = ""
    if "book_author" not in st.session_state:
        st.session_state.book_author = ""
    if "book_isbn" not in st.session_state:
        st.session_state.book_isbn = ""
    if "book_start_date" not in st.session_state:
        st.session_state.book_start_date = None
    if "book_finish_date" not in st.session_state:
        st.session_state.book_finish_date = None
    if "book_index_id" not in st.session_state:
        st.session_state.book_index_id = ""
    if "_index_seq_map" not in st.session_state:
        # Keeps per (YYYYMM) sequence numbers in this session
        st.session_state._index_seq_map = {}
    if "_book_index_assigned" not in st.session_state:
        st.session_state._book_index_assigned = False
    if "_book_index_month_key" not in st.session_state:
        st.session_state._book_index_month_key = None
    if "_prev_form_ready" not in st.session_state:
        st.session_state._prev_form_ready = False
    if "supabase_url" not in st.session_state:
        st.session_state.supabase_url = "https://qqkkygzogkerdixzratb.supabase.co"
    if "supabase_key" not in st.session_state:
        st.session_state.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFxa2t5Z3pvZ2tlcmRpeHpyYXRiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA5NzU1MzgsImV4cCI6MjA3NjU1MTUzOH0.Fj2T-MQokUnRrBLvgDyT_E62AFvEjPdPUd7W5fSW2HA"
    if "user" not in st.session_state:
        st.session_state.user = None
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"  # "login" or "signup"
    if "book_id" not in st.session_state:
        st.session_state.book_id = None


def get_supabase_client() -> Any:
    url = st.session_state.supabase_url
    key = st.session_state.supabase_key
    if not url or not key:
        st.error("Supabase URL or key is missing.")
        return None
    if create_client is None:
        st.error("Supabase package not installed. Run: pip install supabase")
        return None
    try:
        client: Any = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"Failed to create Supabase client: {e}")
        return None


def sign_up(email: str, password: str) -> bool:
    """Sign up a new user"""
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        if response.user:
            st.session_state.user = response.user
            return True
        else:
            st.error("Sign up failed. Please try again.")
            return False
    except Exception as e:
        st.error(f"Sign up error: {e}")
        return False


def sign_in(email: str, password: str) -> bool:
    """Sign in an existing user"""
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if response.user:
            st.session_state.user = response.user
            # Check if user is admin
            if email == "ckleenok@gmail.com":
                st.session_state.is_admin = True
                st.success("🔑 Admin mode activated!")
            else:
                st.session_state.is_admin = False
            return True
        else:
            st.error("Sign in failed. Please check your credentials.")
            return False
    except Exception as e:
        st.error(f"Sign in error: {e}")
        return False


def sign_out():
    """Sign out the current user"""
    supabase = get_supabase_client()
    if supabase and st.session_state.user:
        try:
            supabase.auth.sign_out()
        except:
            pass
    st.session_state.user = None
    st.session_state.is_admin = False
    # Clear all book-related data
    st.session_state.book_id = None
    st.session_state.book_title = ""
    st.session_state.book_author = ""
    st.session_state.book_isbn = ""
    st.session_state.book_start_date = None
    st.session_state.book_finish_date = None
    st.session_state.book_index_id = ""
    st.session_state.original_book_title = ""
    st.session_state.input_text = ""
    st.session_state.mindmap_html = ""


def lookup_book_by_isbn(isbn: str) -> dict:
    """Look up book details by ISBN using multiple APIs (Korean sources first, then international)"""
    import requests
    import re
    
    # Clean ISBN (remove spaces, hyphens)
    clean_isbn = re.sub(r'[-\s]', '', isbn)
    
    # Check if it's a Korean ISBN (starts with 97889)
    is_korean = clean_isbn.startswith('97889')
    
    # For Korean books, try Korean sources first
    if is_korean:
        korean_result = _lookup_korean_books(clean_isbn)
        if korean_result.get('found'):
            return korean_result
    
    # Try Google Books API (works for some Korean books too)
    google_result = _lookup_google_books(clean_isbn)
    if google_result.get('found'):
        return google_result
    
    # Fallback to Open Library API
    openlib_result = _lookup_open_library(clean_isbn)
    if openlib_result.get('found'):
        return openlib_result
    
    # If all fail, return the most informative error
    if is_korean and korean_result.get('error'):
        return korean_result
    elif google_result.get('error'):
        return google_result
    else:
        return openlib_result


def _lookup_google_books(isbn: str) -> dict:
    """Look up book details using Google Books API"""
    import requests
    
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('totalItems', 0) > 0:
                book = data['items'][0]['volumeInfo']
                
                title = book.get('title', '')
                authors = book.get('authors', [])
                author = ', '.join(authors) if authors else ''
                
                # Get ISBN from industryIdentifiers
                isbn_13 = ''
                isbn_10 = ''
                for identifier in book.get('industryIdentifiers', []):
                    if identifier.get('type') == 'ISBN_13':
                        isbn_13 = identifier.get('identifier', '')
                    elif identifier.get('type') == 'ISBN_10':
                        isbn_10 = identifier.get('identifier', '')
                
                # Prefer ISBN-13, fallback to ISBN-10
                final_isbn = isbn_13 or isbn_10 or isbn
                
                return {
                    'title': title,
                    'author': author,
                    'isbn': final_isbn,
                    'found': True,
                    'source': 'Google Books'
                }
            else:
                return {'found': False, 'error': 'Book not found in Google Books'}
        else:
            return {'found': False, 'error': f'Google Books API error: {response.status_code}'}
            
    except Exception as e:
        return {'found': False, 'error': f'Google Books lookup failed: {str(e)}'}


def _lookup_open_library(isbn: str) -> dict:
    """Look up book details using Open Library API"""
    import requests
    
    try:
        url = f"https://openlibrary.org/isbn/{isbn}.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract book information
            title = data.get('title', '')
            authors = data.get('authors', [])
            author_names = []
            
            # Get author names
            for author in authors:
                if isinstance(author, dict) and 'key' in author:
                    author_key = author['key']
                    author_url = f"https://openlibrary.org{author_key}.json"
                    try:
                        author_response = requests.get(author_url, timeout=5)
                        if author_response.status_code == 200:
                            author_data = author_response.json()
                            author_name = author_data.get('name', '')
                            if author_name:
                                author_names.append(author_name)
                    except:
                        pass
            
            return {
                'title': title,
                'author': ', '.join(author_names) if author_names else '',
                'isbn': isbn,
                'found': True,
                'source': 'Open Library'
            }
        else:
            return {'found': False, 'error': 'Book not found in Open Library'}
            
    except Exception as e:
        return {'found': False, 'error': f'Open Library lookup failed: {str(e)}'}


def _lookup_korean_books(isbn: str) -> dict:
    """Look up Korean book details using various Korean sources"""
    import requests
    import re
    
    try:
        # Try web scraping from Korean book sites
        web_result = _lookup_korean_web(isbn)
        if web_result.get('found'):
            return web_result
        
        # Try Aladin API (Korean book database)
        aladin_result = _lookup_aladin(isbn)
        if aladin_result.get('found'):
            return aladin_result
        
        # Try Kyobo Book API
        kyobo_result = _lookup_kyobo(isbn)
        if kyobo_result.get('found'):
            return kyobo_result
        
        # If no Korean sources work, provide helpful message
        return {
            'found': False, 
            'error': f'Korean ISBN {isbn} not found in Korean databases. Please enter book details manually.',
            'suggestion': 'Korean books may not be available in online databases. Please fill in the title and author manually.'
        }
    except Exception as e:
        return {'found': False, 'error': f'Korean book lookup failed: {str(e)}'}


def _lookup_korean_web(isbn: str) -> dict:
    """Look up Korean book details using web scraping"""
    import requests
    from bs4 import BeautifulSoup
    import re
    
    try:
        # Try searching on Aladin website
        search_url = f"https://www.aladin.co.kr/search/wsearchresult.aspx?SearchTarget=All&KeyWord={isbn}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for book title and author in the search results
            title_element = soup.find('a', class_='bo3')
            if title_element:
                title = title_element.get_text().strip()
                
                # Try multiple methods to find author information
                author = ''
                
                # Method 1: Look for author in bo4 class
                author_element = soup.find('a', class_='bo4')
                if author_element:
                    author = author_element.get_text().strip()
                
                # Method 2: Look for author in the book item container
                if not author:
                    book_item = title_element.find_parent('div', class_='ss_book_box')
                    if book_item:
                        # Look for author in various possible locations
                        author_selectors = [
                            'a.bo4',
                            '.ss_book_list1 a',
                            '.ss_book_list2 a',
                            'span[class*="author"]',
                            'div[class*="author"]'
                        ]
                        
                        for selector in author_selectors:
                            author_elem = book_item.select_one(selector)
                            if author_elem and author_elem.get_text().strip():
                                author = author_elem.get_text().strip()
                                break
                
                # Method 3: Look in the text content for author patterns
                if not author:
                    book_item = title_element.find_parent('div', class_='ss_book_box')
                    if book_item:
                        text_content = book_item.get_text()
                        # Look for patterns like "저자: 김철수" or "지은이: 김철수"
                        author_patterns = [
                            r'저자[:\s]*([^,\n]+)',
                            r'지은이[:\s]*([^,\n]+)',
                            r'작가[:\s]*([^,\n]+)',
                            r'글[:\s]*([^,\n]+)'
                        ]
                        
                        for pattern in author_patterns:
                            match = re.search(pattern, text_content)
                            if match:
                                author = match.group(1).strip()
                                break
                
                # Method 4: Try to find author in the book details link
                if not author:
                    book_link = title_element.get('href')
                    if book_link and 'ItemId' in book_link:
                        # Try to get author from the book detail page
                        detail_result = _get_book_details_from_aladin(book_link)
                        if detail_result.get('author'):
                            author = detail_result['author']
                
                return {
                    'title': title,
                    'author': author,
                    'isbn': isbn,
                    'found': True,
                    'source': 'Aladin Website'
                }
        
        return {
            'found': False,
            'error': 'No results found on Aladin website',
            'suggestion': 'Try searching manually on Aladin.co.kr'
        }
        
    except Exception as e:
        return {
            'found': False,
            'error': f'Web scraping failed: {str(e)}',
            'suggestion': 'Please enter book details manually'
        }


def _get_book_details_from_aladin(book_url: str) -> dict:
    """Get detailed book information from Aladin book page"""
    import requests
    from bs4 import BeautifulSoup
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(book_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for author in various possible locations on the detail page
            author_selectors = [
                'a[href*="SearchAuthor"]',
                '.Ere_Sub_Title a',
                '.Ere_sub_title a',
                'span[class*="author"]',
                'div[class*="author"]'
            ]
            
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem and author_elem.get_text().strip():
                    return {'author': author_elem.get_text().strip()}
            
            return {'author': ''}
        
        return {'author': ''}
        
    except Exception as e:
        return {'author': ''}


def _lookup_aladin(isbn: str) -> dict:
    """Look up book details using Aladin API (Korean book database)"""
    import requests
    
    try:
        # Aladin OpenAPI (requires API key, but we can try without)
        # For now, we'll simulate a search that might work
        url = f"http://www.aladin.co.kr/ttb/api/ItemSearch.aspx?ttbkey=YOUR_API_KEY&Query={isbn}&QueryType=ISBN&Output=JS&Version=20131101"
        
        # Since we don't have API key, we'll return a helpful message
        return {
            'found': False,
            'error': 'Aladin API requires authentication',
            'suggestion': 'Try searching manually on Aladin.co.kr or enter book details manually'
        }
    except Exception as e:
        return {'found': False, 'error': f'Aladin lookup failed: {str(e)}'}


def _lookup_kyobo(isbn: str) -> dict:
    """Look up book details using Kyobo Book API"""
    import requests
    
    try:
        # Kyobo Book API (also requires authentication)
        # For now, we'll return a helpful message
        return {
            'found': False,
            'error': 'Kyobo API requires authentication',
            'suggestion': 'Try searching manually on Kyobobook.co.kr or enter book details manually'
        }
    except Exception as e:
        return {'found': False, 'error': f'Kyobo lookup failed: {str(e)}'}


def save_book_to_supabase() -> None:
    client = get_supabase_client()
    if client is None:
        return
    
    # Check if user is authenticated
    if not st.session_state.user:
        st.error("User not authenticated. Please log in again.")
        return
    
    title = (st.session_state.book_title or "").strip()
    author = (st.session_state.book_author or "").strip()
    start_date_val = st.session_state.book_start_date
    finish_date_val = st.session_state.book_finish_date
    
    if not title:
        st.error("Title is required.")
        return
    
    # Check if this is a new book (no existing book_id) or updating existing book
    is_new_book = not st.session_state.get("book_id")
    
    if is_new_book:
        # Generate new index for new book, ensuring uniqueness
        today = date.today()
        base_date = start_date_val or today
        new_index_id = _generate_unique_index_id(base_date, client)
        st.session_state.book_index_id = new_index_id
    
    index_id = st.session_state.book_index_id or ""
    
    payload = {
        "user_id": st.session_state.user.id,
        "user_email": st.session_state.user.email,
        "title": title,
        "author": author or None,
        "isbn": st.session_state.book_isbn or None,
        "start_date": str(start_date_val) if start_date_val else None,
        "finish_date": str(finish_date_val) if finish_date_val else None,
        "index_id": index_id or None,
    }
    
    try:
        if is_new_book:
            # Insert new book
            resp = client.table("books").insert(payload).execute()
            if resp.data and len(resp.data) > 0:
                st.session_state.book_id = resp.data[0]["id"]
                st.session_state.original_book_title = title  # Store for future comparison
                st.success("New book created!")
            else:
                st.error("Failed to create new book.")
        else:
            # Update existing book
            if st.session_state.get("book_id"):
                client.table("books").update(payload).eq("id", st.session_state.book_id).execute()
                st.success("Book updated.")
            else:
                st.error("No book ID found for update.")
    except Exception as e:
        error_msg = str(e)
        if "row-level security policy" in error_msg:
            st.error("RLS Policy Error: User authentication issue. Please try logging out and logging in again.")
            st.info("💡 **Troubleshooting:** If the problem persists, the database may need RLS to be temporarily disabled for testing.")
        else:
            st.error(f"Supabase error saving book: {e}")


def save_summary_to_supabase(content: str) -> None:
    client = get_supabase_client()
    if client is None:
        return
    if not st.session_state.book_id:
        st.error("Please save the book first.")
        return
    if not content.strip():
        st.error("Summary content is empty.")
        return
    
    # Clean up content - remove everything after '-<' or '- <'
    cleaned_content = content.strip()
    if '-<' in cleaned_content:
        cleaned_content = cleaned_content.split('-<')[0].strip()
    elif '- <' in cleaned_content:
        cleaned_content = cleaned_content.split('- <')[0].strip()
    
    if not cleaned_content:
        st.error("No content to save after cleanup.")
        return
    
    payload = {
        "book_id": st.session_state.book_id,
        "content": cleaned_content,
        "is_manual": True,
    }
    try:
        client.table("summaries").insert(payload).execute()
        st.success("Summary saved.")
        
        # Auto-regenerate IAI Tree if one exists
        auto_regenerate_iai_tree()
        
    except Exception as e:
        st.error(f"Supabase error saving summary: {e}")


def auto_regenerate_iai_tree() -> None:
    """Automatically regenerate IAI Tree when new entries are added"""
    if not st.session_state.get("book_id"):
        return
    
    # Check if there's an existing IAI Tree for this book
    client = get_supabase_client()
    if client is None:
        return
    
    try:
        existing_trees = client.table("iai_trees").select("id").eq("book_id", st.session_state.book_id).eq("user_id", st.session_state.user.id).execute()
        
        if existing_trees.data:
            # There's an existing tree, regenerate it
            summaries = load_summaries_for_book(st.session_state.book_id)
            texts = [summary.get("content", "").strip() for summary in summaries if summary.get("content", "").strip()]
            
            if len(texts) > 0:
                # Use default settings for auto-regeneration
                result = cluster_texts(texts, 0.5, 2, 8)  # Default settings
                names = derive_cluster_names(texts, np.array(result["labels"]))
                html = build_iai_tree(texts, np.array(result["labels"]), names)
                
                # Update the existing tree
                book_title = st.session_state.get("book_title", "Untitled Book")
                tree_title = f"IAI Tree - {book_title}"
                
                # Update the tree in database
                client.table("iai_trees").update({
                    "title": tree_title,
                    "html_content": html
                }).eq("book_id", st.session_state.book_id).eq("user_id", st.session_state.user.id).execute()
                
                # Update session state to show the new tree
                st.session_state.mindmap_html = html
                
                st.info("🔄 IAI Tree automatically updated with new entries!")
    except Exception as e:
        # Silently fail for auto-regeneration to avoid disrupting the main flow
        pass


def save_iai_tree_to_supabase(title: str, html_content: str) -> bool:
    """Save IAI Tree to Supabase - automatically overwrites existing trees for the same book"""
    if not st.session_state.get("book_id"):
        st.error("No book selected. Please save book information first.")
        return False
    
    if not st.session_state.get("user"):
        st.error("User not authenticated.")
        return False
    
    client = get_supabase_client()
    if client is None:
        st.error("Supabase is not configured.")
        return False
    
    try:
        # First, check if there's an existing IAI Tree for this book
        existing_trees = client.table("iai_trees").select("id").eq("book_id", st.session_state.book_id).eq("user_id", st.session_state.user.id).execute()
        
        if existing_trees.data:
            # Update existing tree instead of creating new one
            response = client.table("iai_trees").update({
                "title": title,
                "html_content": html_content
            }).eq("book_id", st.session_state.book_id).eq("user_id", st.session_state.user.id).execute()
            
            if response.data:
                st.success("IAI Tree updated successfully!")
                return True
            else:
                st.error("Failed to update IAI Tree.")
                return False
        else:
            # Create new tree if none exists
            response = client.table("iai_trees").insert({
                "book_id": st.session_state.book_id,
                "user_id": st.session_state.user.id,
                "title": title,
                "html_content": html_content
            }).execute()
            
            if response.data:
                st.success("IAI Tree created successfully!")
                return True
            else:
                st.error("Failed to create IAI Tree.")
                return False
    except Exception as e:
        st.error(f"Supabase error saving IAI Tree: {e}")
        return False


def _compute_index_id(d: date, seq: int) -> str:
    y = d.year
    m = d.month
    return f"{y}/{m:02d}/{seq:03d}"


def _get_seq_for_month(d: date) -> int:
    key = f"{d.year}{d.month:02d}"
    if key not in st.session_state._index_seq_map:
        st.session_state._index_seq_map[key] = 1
    return st.session_state._index_seq_map[key]


def _set_seq_for_month(d: date, seq: int) -> None:
    key = f"{d.year}{d.month:02d}"
    st.session_state._index_seq_map[key] = max(1, int(seq))


def _increment_seq_for_month(d: date) -> int:
    current = _get_seq_for_month(d)
    _set_seq_for_month(d, current + 1)
    return current + 1


def _generate_unique_index_id(base_date: date, client: Any) -> str:
    """Generate a unique index ID by checking existing ones in the database for current user"""
    month_key = f"{base_date.year}{base_date.month:02d}"
    seq = 1
    
    while True:
        index_id = _compute_index_id(base_date, seq)
        try:
            # Check if this index_id already exists for current user
            existing = client.table("books").select("id").eq("index_id", index_id).eq("user_id", st.session_state.user.id).execute()
            if not existing.data:
                # This index_id is available for current user
                return index_id
            else:
                # This index_id exists for current user, try next sequence
                seq += 1
        except Exception:
            # If there's an error checking, just return the generated ID
            return index_id


def parse_lines_to_items(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    # Filter out empty lines and common separator patterns
    separator_patterns = [
        r'^-+$',  # ---, ----, etc.
        r'^=+$',  # ===, ====, etc.
        r'^\*+$',  # ***, ****, etc.
        r'^_+$',   # ___, ____, etc.
    ]
    items = []
    for ln in lines:
        if len(ln) > 0:
            # Check if line matches any separator pattern
            is_separator = any(re.match(pattern, ln) for pattern in separator_patterns)
            if not is_separator:
                items.append(ln)
    return items


def estimate_k(n: int, grouping_strength: float, k_min: int, k_max: int) -> int:
    if n <= 2:
        return 1
    base = max(2, int(round(math.sqrt(n))))
    min_k = max(1, min(k_min, n))
    max_k = max(min_k, min(k_max, n))
    k = int(round(min_k + grouping_strength * (max_k - min_k)))
    k = max(min_k, min(k, max_k))
    # Nudge toward base for natural grouping
    k = int(round(0.5 * k + 0.5 * base))
    k = max(min_k, min(k, max_k))
    return k


def cluster_texts(texts: List[str], grouping_strength: float, k_min: int, k_max: int) -> Dict[str, Any]:
    """Enhanced AI-powered text clustering with improved semantic analysis"""
    if len(texts) == 0:
        return {"labels": [], "k": 0, "centers": None, "score": None}

    k = estimate_k(len(texts), grouping_strength, k_min, k_max)

    # Enhanced vectorization with better parameters for semantic understanding
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        max_df=0.8,  # Reduced to focus on more specific terms
        min_df=1,
        ngram_range=(1, 3),  # Include trigrams for better context
        max_features=500,  # Increased for richer feature space
        sublinear_tf=True,  # Use sublinear tf scaling
    )
    
    # Enhanced preprocessing pipeline
    pipeline = make_pipeline(
        vectorizer, 
        Normalizer(copy=False, norm='l2')  # L2 normalization for better clustering
    )
    X = pipeline.fit_transform(texts)

    # Guard rails: K cannot exceed samples
    k = max(1, min(k, X.shape[0]))
    if k == 1:
        labels = np.zeros(X.shape[0], dtype=int)
        return {"labels": labels, "k": 1, "centers": None, "score": None}

    # Enhanced KMeans with better initialization and more iterations
    model = KMeans(
        n_clusters=k, 
        n_init=20,  # Increased for better initialization
        random_state=42,
        max_iter=500,  # More iterations for convergence
        algorithm='lloyd'  # Explicit algorithm specification
    )
    labels = model.fit_predict(X)

    score = None
    try:
        if k > 1 and X.shape[0] > k:
            # Enhanced scoring with multiple metrics
            score = silhouette_score(X, labels)
    except Exception:
        score = None

    return {"labels": labels, "k": k, "centers": getattr(model, "cluster_centers_", None), "score": score}


def derive_cluster_names(texts: List[str], labels: np.ndarray, top_n: int = 3) -> Dict[int, str]:
    """AI-powered cluster naming with Insight-Action-Integration framework"""
    df = pd.DataFrame({"text": texts, "label": labels})
    names: Dict[int, str] = {}
    
    # Define semantic categories for the IAI framework (Korean + English)
    insight_keywords = [
        # English
        'insight', 'realization', 'understanding', 'awareness', 'epiphany', 'revelation', 
        'discovery', 'learning', 'lesson', 'perspective', 'viewpoint', 'mindset', 'belief', 
        'thought', 'idea', 'concept', 'principle', 'philosophy', 'wisdom', 'knowledge',
        'realize', 'understand', 'learn', 'discover', 'realize', 'comprehend',
        # Korean
        '통찰', '깨달음', '이해', '인식', '깨우침', '발견', '학습', '교훈', '관점', '사고', 
        '생각', '아이디어', '개념', '원리', '철학', '지혜', '지식', '알다', '깨닫다', 
        '이해하다', '배우다', '발견하다', '인식하다', '깨우치다', '알아차리다'
    ]
    
    action_keywords = [
        # English
        'action', 'do', 'implement', 'practice', 'apply', 'execute', 'start', 'begin', 
        'try', 'attempt', 'experiment', 'test', 'use', 'utilize', 'adopt', 'change', 
        'modify', 'improve', 'enhance', 'develop', 'build', 'create', 'make', 'achieve', 
        'accomplish', 'goal', 'plan', 'strategy', 'method', 'technique', 'approach', 
        'way', 'how', 'what', 'when', 'where', 'will', 'should', 'must', 'need',
        # Korean
        '행동', '실행', '실천', '적용', '시작', '시도', '실험', '테스트', '사용', '활용', 
        '채택', '변화', '개선', '향상', '개발', '구축', '창조', '만들다', '달성', '목표', 
        '계획', '전략', '방법', '기법', '접근', '방식', '어떻게', '무엇을', '언제', '어디서',
        '할', '해야', '해야만', '필요', '시작하다', '실행하다', '적용하다', '시도하다'
    ]
    
    integration_keywords = [
        # English
        'habit', 'routine', 'system', 'process', 'workflow', 'practice', 'discipline', 
        'consistency', 'regular', 'daily', 'weekly', 'monthly', 'schedule', 'time', 
        'environment', 'setup', 'structure', 'framework', 'methodology', 'approach', 
        'lifestyle', 'behavior', 'pattern', 'tradition', 'ritual', 'ceremony', 'custom',
        'always', 'every', 'often', 'frequently', 'consistently', 'regularly',
        # Korean
        '습관', '루틴', '시스템', '과정', '워크플로우', '연습', '훈련', '일관성', '정기적', 
        '일상', '주간', '월간', '스케줄', '시간', '환경', '설정', '구조', '프레임워크', 
        '방법론', '라이프스타일', '행동', '패턴', '전통', '의식', '관례', '항상', '매일', 
        '자주', '꾸준히', '정기적으로', '습관화', '체계화', '내면화'
    ]
    
    # Track category distribution to ensure balanced allocation
    category_counts = {"insight": 0, "action": 0, "integration": 0}
    total_clusters = len(df["label"].unique())
    
    for label, group in df.groupby("label"):
        # Analyze text content to determine category
        all_text = " ".join(group["text"].tolist()).lower()
        
        # Count keyword matches for each category
        insight_score = sum(1 for keyword in insight_keywords if keyword in all_text)
        action_score = sum(1 for keyword in action_keywords if keyword in all_text)
        integration_score = sum(1 for keyword in integration_keywords if keyword in all_text)
        
        # Enhanced scoring: also consider text length and content patterns
        text_length = len(all_text)
        
        # Boost scores based on content patterns
        if any(word in all_text for word in ['해야', '해야만', '필요', '시작', '실행', '적용']):
            action_score += 2
        if any(word in all_text for word in ['습관', '루틴', '매일', '정기', '체계']):
            integration_score += 2
        if any(word in all_text for word in ['깨달음', '이해', '알다', '인식', '발견']):
            insight_score += 2
        
        # Debug: Print scores for each cluster
        print(f"Cluster {label}: Insight={insight_score}, Action={action_score}, Integration={integration_score}")
        print(f"Text sample: {all_text[:100]}...")
        
        # Determine category based on highest score
        # If all scores are 0, use balanced distribution
        if insight_score == 0 and action_score == 0 and integration_score == 0:
            # Use balanced distribution based on current category counts
            if category_counts["insight"] <= category_counts["action"] and category_counts["insight"] <= category_counts["integration"]:
                category = "💡 통찰 (Insight)"
                category_counts["insight"] += 1
            elif category_counts["action"] <= category_counts["integration"]:
                category = "🎯 행동 (Action)"
                category_counts["action"] += 1
            else:
                category = "🔄 내면화 (Integration)"
                category_counts["integration"] += 1
        elif insight_score >= action_score and insight_score >= integration_score:
            category = "💡 통찰 (Insight)"
            category_counts["insight"] += 1
        elif action_score >= integration_score:
            category = "🎯 행동 (Action)"
            category_counts["action"] += 1
        else:
            category = "🔄 내면화 (Integration)"
            category_counts["integration"] += 1
        
        # Extract meaningful terms from the cluster
        tokens: Dict[str, int] = {}
        for t in group["text"].tolist():
            text = str(t).lower().replace("\n", " ").replace("\t", " ")
            words = [w.strip() for w in text.split() if w.strip()]
            
            for w in words:
                clean_w = "".join([c for c in w if c.isalnum() or c == '-'])
                if len(clean_w) <= 2 or clean_w in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'oil', 'sit', 'sun', 'try', 'use', 'why']:
                    continue
                tokens[clean_w] = tokens.get(clean_w, 0) + 1
        
        # Get top terms for this cluster
        def score_token(token_freq):
            token, freq = token_freq
            length_bonus = min(len(token) * 0.1, 2.0)
            return freq + length_bonus
        
        top = sorted(tokens.items(), key=lambda x: (-score_token(x), x[0]))[:2]
        
        if top:
            top_words = [w for w, _ in top]
            if len(top_words) >= 2:
                names[label] = f"{category}: {top_words[0].title()} & {top_words[1].title()}"
            else:
                names[label] = f"{category}: {top_words[0].title()}"
        else:
            names[label] = category
    
    # Print final distribution for debugging
    print(f"Final distribution: Insight={category_counts['insight']}, Action={category_counts['action']}, Integration={category_counts['integration']}")
    
    return names


def build_iai_tree(texts: List[str], labels: np.ndarray, cluster_names: Dict[int, str]) -> str:
    """Build IAI Framework tree structure instead of mind map"""
    
    # Group texts by their labels
    clusters = {}
    for idx, (text, label) in enumerate(zip(texts, labels)):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((idx, text))
    
    # Create HTML tree structure
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>IAI Framework Tree</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 30px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 3px solid #2c3e50;
            }
            .header h1 {
                color: #2c3e50;
                margin: 0;
                font-size: 28px;
            }
            .header p {
                color: #7f8c8d;
                margin: 10px 0 0 0;
                font-size: 16px;
            }
            .tree {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .category {
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .category-header {
                padding: 15px 20px;
                color: white;
                font-weight: bold;
                font-size: 18px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .category-content {
                background: #f8f9fa;
                padding: 20px;
            }
            .insight {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
            }
            .action {
                background: linear-gradient(135deg, #f39c12, #e67e22);
            }
            .integration {
                background: linear-gradient(135deg, #27ae60, #229954);
            }
            .item {
                background: white;
                margin: 8px 0;
                padding: 12px 16px;
                border-radius: 6px;
                border-left: 4px solid #bdc3c7;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            .item:hover {
                transform: translateX(5px);
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
            }
            .item.insight-item {
                border-left-color: #e74c3c;
            }
            .item.action-item {
                border-left-color: #f39c12;
            }
            .item.integration-item {
                border-left-color: #27ae60;
            }
            .item-number {
                color: #7f8c8d;
                font-weight: bold;
                margin-right: 8px;
            }
            .item-text {
                color: #2c3e50;
                line-height: 1.5;
            }
            .stats {
                text-align: center;
                margin-top: 20px;
                padding: 15px;
                background: #ecf0f1;
                border-radius: 6px;
                color: #7f8c8d;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 IAI Framework Tree</h1>
                <p>Insight - Action - Integration 구조로 정리된 학습 내용</p>
            </div>
            <div class="tree">
    """
    
    # Define category order and styling
    category_order = [
        ("💡 통찰 (Insight)", "insight", "#e74c3c"),
        ("🎯 행동 (Action)", "action", "#f39c12"),
        ("🔄 내면화 (Integration)", "integration", "#27ae60")
    ]
    
    # Process each category
    for category_name, category_class, category_color in category_order:
        # Find clusters that belong to this category
        category_clusters = []
        for label, cluster_name in cluster_names.items():
            if category_name.split()[0] in cluster_name:  # Check for emoji match
                category_clusters.append((label, cluster_name, clusters.get(label, [])))
        
        if category_clusters:
            html_content += f"""
                <div class="category">
                    <div class="category-header {category_class}">
                        <span>{category_name}</span>
                    </div>
                    <div class="category-content">
            """
            
            for label, cluster_name, items in category_clusters:
                if items:
                    for idx, text in items:
                        html_content += f"""
                        <div class="item {category_class}-item">
                            <span class="item-number">{idx + 1}.</span>
                            <span class="item-text">{text}</span>
                        </div>
                        """
            
            html_content += """
                    </div>
                </div>
            """
    
    # Add statistics
    total_items = len(texts)
    html_content += f"""
            </div>
            <div class="stats">
                📊 총 {total_items}개 항목이 IAI 프레임워크로 구조화되었습니다
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


def render_header() -> None:
    st.title(APP_TITLE)
    st.caption("Use Quick Add (Enter or Add button) to append items, or edit the list below. Then tap 'IAI Tree Creation'.")


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Book")
    
    # ISBN lookup section
    st.sidebar.markdown("**📚 ISBN Lookup**")
    # Use a unique key for ISBN input to prevent caching and always initialize
    import uuid
    isbn_key = f"isbn_input_{uuid.uuid4().hex[:8]}"
    isbn_input = st.sidebar.text_input("ISBN", placeholder="Enter ISBN", key=isbn_key, value="")
    
    if st.sidebar.button("🔍 Lookup Book", use_container_width=True):
        if isbn_input:
            with st.spinner("Looking up book information..."):
                book_info = lookup_book_by_isbn(isbn_input)
                if book_info.get('found'):
                    st.session_state.book_title = book_info.get('title', '')
                    st.session_state.book_author = book_info.get('author', '')
                    st.session_state.book_isbn = book_info.get('isbn', '')
                    source = book_info.get('source', 'Unknown source')
                    
                    # Show what was found
                    title = book_info.get('title', '')
                    author = book_info.get('author', '')
                    
                    if title and author:
                        st.sidebar.success(f"📚 Book information loaded from {source}!")
                        st.sidebar.info(f"**Title:** {title}")
                        st.sidebar.info(f"**Author:** {author}")
                    elif title:
                        st.sidebar.success(f"📚 Book title loaded from {source}!")
                        st.sidebar.info(f"**Title:** {title}")
                        st.sidebar.warning("⚠️ Author not found. Please enter manually.")
                    else:
                        st.sidebar.success(f"📚 Book information loaded from {source}!")
                    
                    # Clear the ISBN input field after successful lookup
                    st.rerun()
                    
                else:
                    error_msg = book_info.get('error', 'Unknown error')
                    suggestion = book_info.get('suggestion', '')
                    
                    st.sidebar.error(f"Book not found: {error_msg}")
                    if suggestion:
                        st.sidebar.info(f"💡 {suggestion}")
        else:
            st.sidebar.error("Please enter an ISBN")
    
    st.sidebar.markdown("---")
    
    # Manual input fields
    if st.session_state.get("book_id"):
        st.sidebar.markdown("**✏️ Edit Book Details**")
        st.sidebar.info("📝 You are editing an existing book")
    else:
        st.sidebar.markdown("**📝 Book Details**")
    
    st.session_state.book_title = st.sidebar.text_input("Title", value=st.session_state.book_title)
    st.session_state.book_author = st.sidebar.text_input("Author", value=st.session_state.book_author)
    st.session_state.book_isbn = st.sidebar.text_input("ISBN", value=st.session_state.book_isbn)
    st.session_state.book_start_date = st.sidebar.date_input("Start date", value=st.session_state.book_start_date)
    st.session_state.book_finish_date = st.sidebar.date_input("Finish date", value=st.session_state.book_finish_date)

    # Auto Index (Year/Month/Sequence) based on input order - internal only
    today = date.today()
    base_date = st.session_state.book_start_date or today

    month_key = f"{base_date.year}{base_date.month:02d}"
    form_ready = bool(st.session_state.book_title and base_date)

    # Assign on first time the form becomes ready (order of completion)
    if form_ready and not st.session_state._prev_form_ready:
        seq = _get_seq_for_month(base_date)
        st.session_state.book_index_id = _compute_index_id(base_date, seq)
        st.session_state._book_index_assigned = True
        st.session_state._book_index_month_key = month_key
        _set_seq_for_month(base_date, seq + 1)

    # If month changes after assignment, reassign using new month sequence
    if st.session_state._book_index_assigned and st.session_state._book_index_month_key != month_key:
        seq = _get_seq_for_month(base_date)
        st.session_state.book_index_id = _compute_index_id(base_date, seq)
        st.session_state._book_index_month_key = month_key
        _set_seq_for_month(base_date, seq + 1)

    # Track readiness for next render to detect transitions
    st.session_state._prev_form_ready = form_ready
    
    # Save book button
    save_button_text = "💾 Update Book" if st.session_state.get("book_id") else "💾 Save Book"
    if st.sidebar.button(save_button_text, type="primary"):
        save_book_to_supabase()

    # Return default settings for IAI Tree creation
    return {"grouping_strength": 0.5, "k_min": 2, "k_max": 8}


def render_input_ui() -> None:
    st.subheader("Input")
    
    # Quick add input field
    quick_add_value = st.text_input(
        label="Quick Add (press Enter)",
        key="quick_add",
        placeholder="Type an item and press Enter",
    )
    
    # Add Item button
    if st.button("Add Item", type="secondary", use_container_width=True):
        if quick_add_value and st.session_state.get("book_id"):
            # Clean up pasted content - remove everything after '-<' or '- <'
            cleaned_value = quick_add_value.strip()
            if '-<' in cleaned_value:
                cleaned_value = cleaned_value.split('-<')[0].strip()
            elif '- <' in cleaned_value:
                cleaned_value = cleaned_value.split('- <')[0].strip()
            
            if cleaned_value:
                # Save immediately as individual entry
                save_summary_to_supabase(cleaned_value)
                st.rerun()  # Refresh to clear input and show updated entries
        elif quick_add_value and not st.session_state.get("book_id"):
            st.warning("Please save the book first before adding content.")
    
    # Handle Enter key press
    if quick_add_value and st.session_state.get("book_id"):
        # Check if this is a new input (not from rerun)
        if not hasattr(st.session_state, '_last_quick_add') or st.session_state._last_quick_add != quick_add_value:
            st.session_state._last_quick_add = quick_add_value
            
            # Clean up pasted content - remove everything after '-<' or '- <'
            cleaned_value = quick_add_value.strip()
            if '-<' in cleaned_value:
                cleaned_value = cleaned_value.split('-<')[0].strip()
            elif '- <' in cleaned_value:
                cleaned_value = cleaned_value.split('- <')[0].strip()
            
            if cleaned_value:
                # Save immediately as individual entry
                save_summary_to_supabase(cleaned_value)
                st.rerun()  # Refresh to clear input and show updated entries

    # Show recent entries for this book
    if st.session_state.get("book_id"):
        summaries = load_summaries_for_book(st.session_state.book_id)
        if summaries:
            st.caption("Recent entries")
            # Show only the first 3 entries (most recent ones, since data is already sorted by created_at desc)
            for summary in summaries[:3]:
                content = summary.get("content", "").strip()
                if content:
                    st.write(f"• {content}")


# Removed multi-row helpers in favor of single textarea input


def render_action_bar(settings: Dict[str, Any]) -> None:
    left, right = st.columns([1, 1])
    with left:
        if st.button("IAI Tree Creation", type="primary"):
            # Get all entries for current book for mind map
            if st.session_state.get("book_id"):
                summaries = load_summaries_for_book(st.session_state.book_id)
                texts = [summary.get("content", "").strip() for summary in summaries if summary.get("content", "").strip()]
            else:
                texts = []
            
            with st.spinner("Clustering and building mind map..."):
                result = cluster_texts(texts, settings["grouping_strength"], settings["k_min"], settings["k_max"])
                if len(texts) == 0:
                    st.warning("Please add some entries first.")
                else:
                    names = derive_cluster_names(texts, np.array(result["labels"]))
                    html = build_iai_tree(texts, np.array(result["labels"]), names)
                    st.session_state.mindmap_html = html
    with right:
        if st.session_state.mindmap_html:
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download HTML",
                    data=st.session_state.mindmap_html,
                    file_name="iai_tree.html",
                    mime="text/html",
                )
            with col2:
                if st.button("💾 Save IAI Tree", type="secondary"):
                    # Get book title for the tree title
                    book_title = st.session_state.get("book_title", "Untitled Book")
                    tree_title = f"IAI Tree - {book_title}"
                    
                    if save_iai_tree_to_supabase(tree_title, st.session_state.mindmap_html):
                        st.rerun()


def render_iai_tree() -> None:
    if st.session_state.mindmap_html:
        st.subheader("🧠 IAI Framework Tree")
        components.html(st.session_state.mindmap_html, height=680, scrolling=True)


def load_books_from_supabase() -> List[Dict[str, Any]]:
    client = get_supabase_client()
    if client is None:
        return []
    if not st.session_state.user:
        return []
    try:
        resp = client.table("books").select("*").eq("user_id", st.session_state.user.id).order("created_at", desc=True).execute()
        return resp.data or []
    except Exception as e:
        st.error(f"Failed to load books: {e}")
        return []




def load_iai_trees_for_book(book_id: str) -> List[Dict[str, Any]]:
    """Load IAI Trees for a specific book"""
    client = get_supabase_client()
    if client is None:
        return []
    if not st.session_state.user:
        return []
    try:
        response = client.table("iai_trees").select("*").eq("book_id", book_id).eq("user_id", st.session_state.user.id).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Failed to load IAI Trees for book: {e}")
        return []


def load_summaries_for_book(book_id: str) -> List[Dict[str, Any]]:
    client = get_supabase_client()
    if client is None:
        return []
    try:
        resp = client.table("summaries").select("*").eq("book_id", book_id).order("created_at", desc=True).execute()
        return resp.data or []
    except Exception as e:
        st.error(f"Failed to load summaries: {e}")
        return []


def delete_book_from_supabase(book_id: str) -> bool:
    client = get_supabase_client()
    if client is None:
        return False
    try:
        # Delete summaries first (due to foreign key constraint)
        client.table("summaries").delete().eq("book_id", book_id).execute()
        # Delete book
        client.table("books").delete().eq("id", book_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to delete book: {e}")
        return False


def delete_summary_from_supabase(summary_id: str) -> bool:
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("summaries").delete().eq("id", summary_id).execute()
        
        # Auto-regenerate IAI Tree if one exists
        auto_regenerate_iai_tree()
        
        return True
    except Exception as e:
        st.error(f"Failed to delete summary: {e}")
        return False


def delete_iai_tree_from_supabase(tree_id: str) -> bool:
    """Delete IAI Tree from Supabase"""
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("iai_trees").delete().eq("id", tree_id).execute()
        return True
    except Exception as e:
        st.error(f"Supabase error deleting IAI Tree: {e}")
        return False


def update_summary_in_supabase(summary_id: str, new_content: str) -> bool:
    client = get_supabase_client()
    if client is None:
        return False
    try:
        # Clean up content - remove everything after '-<' or '- <'
        cleaned_content = new_content.strip()
        if '-<' in cleaned_content:
            cleaned_content = cleaned_content.split('-<')[0].strip()
        elif '- <' in cleaned_content:
            cleaned_content = cleaned_content.split('- <')[0].strip()
        
        if not cleaned_content:
            st.error("No content to save after cleanup.")
            return False
            
        client.table("summaries").update({"content": cleaned_content}).eq("id", summary_id).execute()
        
        # Auto-regenerate IAI Tree if one exists
        auto_regenerate_iai_tree()
        
        return True
    except Exception as e:
        st.error(f"Failed to update summary: {e}")
        return False


def render_auth_page():
    """Render authentication page (login/signup)"""
    st.title("📚 Book Organizer")
    st.markdown("---")
    
    # Center the toggle buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Toggle between login and signup
        toggle_col1, toggle_col2 = st.columns(2)
        
        with toggle_col1:
            if st.button("🔑 Login", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()
        
        with toggle_col2:
            if st.button("📝 Sign Up", use_container_width=True):
                st.session_state.auth_page = "signup"
                st.rerun()
    
    st.markdown("---")
    
    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.session_state.auth_page == "login":
            st.subheader("🔑 Login")
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="your@email.com")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if email and password:
                        if sign_in(email, password):
                            st.success("Login successful!")
                            st.rerun()
                    else:
                        st.error("Please fill in all fields.")
        
        else:  # signup
            st.subheader("📝 Sign Up")
            with st.form("signup_form"):
                email = st.text_input("Email", placeholder="your@email.com")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Sign Up", use_container_width=True)
                
                if submit:
                    if email and password and confirm_password:
                        if password == confirm_password:
                            if sign_up(email, password):
                                st.success("Sign up successful! Please check your email to verify your account.")
                                st.rerun()
                        else:
                            st.error("Passwords do not match.")
                    else:
                        st.error("Please fill in all fields.")


def render_admin_dashboard():
    """Render admin dashboard with system statistics and management tools"""
    st.title("🔧 Admin Dashboard")
    
    # Admin info
    st.info(f"👤 Logged in as: {st.session_state.user.email}")
    
    # System statistics
    st.subheader("📊 System Statistics")
    
    try:
        supabase = get_supabase_client()
        
        # Get total books (accessible via RLS)
        books_response = supabase.table("books").select("id").execute()
        total_books = len(books_response.data) if books_response.data else 0
        
        # Get total summaries (accessible via RLS)
        summaries_response = supabase.table("summaries").select("id").execute()
        total_summaries = len(summaries_response.data) if summaries_response.data else 0
        
        # Get unique users from books table (since we can't access auth.users directly)
        unique_users_response = supabase.table("books").select("user_id").execute()
        unique_users = set()
        if unique_users_response.data:
            for book in unique_users_response.data:
                if book.get('user_id'):
                    unique_users.add(book['user_id'])
        total_users = len(unique_users)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Users", total_users)
        with col2:
            st.metric("Total Books", total_books)
        with col3:
            st.metric("Total Summaries", total_summaries)
        
        # User-specific statistics
        st.subheader("👥 User Statistics")
        
        # Get all books with user info
        all_books = supabase.table("books").select("user_id, user_email, title, created_at").execute()
        
        if all_books.data:
            # Group books by user
            user_stats = {}
            for book in all_books.data:
                user_id = book.get('user_id')
                user_email = book.get('user_email', 'Unknown email')
                if user_id:
                    if user_id not in user_stats:
                        user_stats[user_id] = {
                            'email': user_email,
                            'books': 0,
                            'recent_books': []
                        }
                    user_stats[user_id]['books'] += 1
                    user_stats[user_id]['recent_books'].append({
                        'title': book.get('title', 'Unknown'),
                        'created_at': book.get('created_at', '')
                    })
            
            # Sort users by book count
            sorted_users = sorted(user_stats.items(), key=lambda x: x[1]['books'], reverse=True)
            
            # Display user statistics
            for user_id, stats in sorted_users:
                st.write(f"**📧 {stats['email']}**")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Books", stats['books'])
                
                with col2:
                    # Show recent books for this user
                    recent_books = sorted(stats['recent_books'], key=lambda x: x['created_at'], reverse=True)[:3]
                    st.write("Recent books:")
                    for book in recent_books:
                        date_str = book['created_at'][:10] if book['created_at'] else 'Unknown date'
                        st.write(f"• {book['title']} ({date_str})")
                
                st.write("---")
        else:
            st.write("No user data found.")
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        recent_books = supabase.table("books").select("title, author, created_at, user_id").order("created_at", desc=True).limit(10).execute()
        
        if recent_books.data:
            st.write("**Recent Books Added:**")
            for book in recent_books.data:
                st.write(f"• {book.get('title', 'Unknown')} by {book.get('author', 'Unknown')} - {book.get('created_at', '')[:10]}")
        else:
            st.write("No recent activity found.")
            
        # Additional admin info
        st.subheader("ℹ️ Admin Information")
        st.info("""
        **Note:** User statistics are based on active users who have created books. 
        The `auth.users` table is not directly accessible due to security policies.
        """)
            
    except Exception as e:
        st.error(f"Failed to load admin data: {e}")


def render_library_page() -> None:
    st.title("📚 Book Library")
    
    books = load_books_from_supabase()
    
    if not books:
        st.info("No books found. Go to the main page to add your first book!")
        return
    
    # Filters
    st.subheader("🔍 Filters")
    
    # All filters in one row - search field normal, year/month reduced to 40%
    col1, col2, col3, col4 = st.columns([2, 0.8, 0.8, 1.4])
    
    with col1:
        # Text search for title/author
        search_term = st.text_input("Search title or author", placeholder="Type to search...")
    
    with col2:
        # Get unique years from all books
        years = set()
        for book in books:
            start_date = book.get("start_date")
            finish_date = book.get("finish_date")
            if start_date:
                years.add(str(start_date)[:4])
            if finish_date:
                years.add(str(finish_date)[:4])
        
        years = sorted(list(years), reverse=True)
        years.insert(0, "All years")
        selected_year = st.selectbox("Year", years)
    
    with col3:
        # Month filter
        months = [
            "All months", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        selected_month = st.selectbox("Month", months)
    
    # Apply filters
    filtered_books = []
    for book in books:
        # Text search filter
        text_match = True
        if search_term:
            search_lower = search_term.lower()
            title_author_text = (book.get("title", "") + " " + book.get("author", "")).lower()
            text_match = search_lower in title_author_text
        
        # Year filter
        year_match = True
        if selected_year != "All years":
            start_date = book.get("start_date")
            finish_date = book.get("finish_date")
            book_year = None
            if start_date:
                book_year = str(start_date)[:4]
            elif finish_date:
                book_year = str(finish_date)[:4]
            year_match = book_year == selected_year
        
        # Month filter
        month_match = True
        if selected_month != "All months":
            start_date = book.get("start_date")
            finish_date = book.get("finish_date")
            book_month = None
            if start_date:
                try:
                    book_month = datetime.strptime(str(start_date), "%Y-%m-%d").strftime("%B")
                except:
                    pass
            elif finish_date:
                try:
                    book_month = datetime.strptime(str(finish_date), "%Y-%m-%d").strftime("%B")
                except:
                    pass
            month_match = book_month == selected_month
        
        if text_match and year_match and month_match:
            filtered_books.append(book)
    
    books = filtered_books
    
    # Add CSS for book cards
    st.markdown("""
    <style>
    .book-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-height: 300px;
        display: flex;
        flex-direction: column;
        font-size: 1.0em;
    }
    .book-card h3 {
        margin-top: 0;
        margin-bottom: 12px;
        color: #2c3e50;
        font-size: 1.3em;
        line-height: 1.2;
        min-height: 2.4em;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .book-details {
        margin-bottom: 12px;
    }
    .book-details p {
        margin: 4px 0;
        font-size: 1.0em;
        color: #555;
        line-height: 1.2;
    }
    .book-entries {
        margin-bottom: 12px;
    }
    .entry-count {
        font-size: 1.0em;
        color: #666;
        padding: 6px 8px;
        background-color: #f8f9fa;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    .book-actions {
        margin-top: auto;
    }
    .button-row {
        display: flex;
        gap: 8px;
        margin-bottom: 8px;
    }
    .button-row:last-child {
        margin-bottom: 0;
    }
    .action-btn {
        flex: 1;
        font-size: 1.1em;
        padding: 8px 12px;
        min-height: 32px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background-color: #f8f9fa;
        cursor: pointer;
        transition: all 0.2s;
    }
    .action-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .view-btn {
        background-color: #007bff;
        color: white;
        border-color: #007bff;
    }
    .view-btn:hover {
        background-color: #0056b3;
    }
    .edit-btn {
        background-color: #28a745;
        color: white;
        border-color: #28a745;
    }
    .edit-btn:hover {
        background-color: #1e7e34;
    }
    .iai-btn {
        background-color: #17a2b8;
        color: white;
        border-color: #17a2b8;
    }
    .iai-btn:hover {
        background-color: #138496;
    }
    .delete-btn {
        background-color: #dc3545;
        color: white;
        border-color: #dc3545;
    }
    .delete-btn:hover {
        background-color: #c82333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Book cards - display 5 books per row
    for i in range(0, len(books), 5):
        # Create a row with up to 5 books
        book_row = books[i:i+5]
        cols = st.columns(5)
        
        for j, book in enumerate(book_row):
            with cols[j]:
                # Get entry count for this book
                summaries = load_summaries_for_book(book['id'])
                entry_count = len(summaries)
                
                # Create book card with all content inside the white box
                with st.container():
                    # Build finished date HTML conditionally
                    finished_html = ""
                    if book.get("finish_date") and book.get("finish_date") != "None":
                        finished_html = f'<p><strong>Finished:</strong> {book.get("finish_date")}</p>'
                    
                    st.markdown(f"""
                    <div class="book-card">
                        <h3>{book.get("title", "Untitled")}</h3>
                        <div class="book-details">
                            <p><strong>Author:</strong> {book.get('author', 'N/A')}</p>
                            <p><strong>Index:</strong> {book.get('index_id', 'N/A')}</p>
                            <p><strong>Started:</strong> {book.get('start_date', 'N/A')}</p>
                            {finished_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show entry count with expandable preview
                    if entry_count > 0:
                        with st.expander(f"📝 **{entry_count} entries** (click to preview)", expanded=False):
                            # Load and display summaries for preview
                            summaries = load_summaries_for_book(book['id'])
                            if summaries:
                                for k, summary in enumerate(summaries[:5]):  # Show first 5 entries
                                    created_at = summary.get('created_at', '')
                                    time_str = created_at.split('T')[1][:5] if 'T' in created_at else created_at[:5]
                                    content = summary.get('content', '')
                                    # Truncate long content
                                    preview_content = content[:100] + "..." if len(content) > 100 else content
                                    st.write(f"**{time_str}:** {preview_content}")
                                
                                if len(summaries) > 5:
                                    st.write(f"... and {len(summaries) - 5} more entries")
                            else:
                                st.write("No entries found")
                    else:
                        st.write("📝 **No entries yet**")
                    
                    # Create 2x2 button layout using Streamlit columns
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📖 View", key=f"view_{book['id']}", use_container_width=True):
                            st.session_state.selected_book_id = book['id']
                            st.session_state.current_page = "book_detail"
                            st.rerun()
                    
                    with col2:
                        if st.button(f"✏️ Edit", key=f"edit_{book['id']}", use_container_width=True):
                            st.session_state.selected_book_id = book['id']
                            st.session_state.current_page = "main"
                            # Load book data into form
                            st.session_state.book_title = book.get("title", "")
                            st.session_state.book_author = book.get("author", "")
                            st.session_state.book_start_date = book.get("start_date")
                            st.session_state.book_finish_date = book.get("finish_date")
                            st.session_state.book_index_id = book.get("index_id", "")
                            st.session_state.book_id = book['id']
                            # Store original title for comparison
                            st.session_state.original_book_title = book.get("title", "")
                            
                            # Clear input text when editing (don't load existing content)
                            st.session_state.input_text = ""
                            
                            st.rerun()
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        # Check if there are IAI Trees for this book
                        iai_trees = load_iai_trees_for_book(book['id'])
                        if iai_trees:
                            if st.button(f"🌳 IAI Tree", key=f"iai_tree_{book['id']}", use_container_width=True):
                                st.session_state.selected_book_id = book['id']
                                st.session_state.current_page = "book_iai_trees"
                                st.rerun()
                        else:
                            st.button(f"🌳 IAI Tree", key=f"iai_tree_{book['id']}", use_container_width=True, disabled=True)
                    
                    with col4:
                        if st.button(f"🗑️ Delete", key=f"delete_{book['id']}", type="secondary", use_container_width=True):
                            # Confirm deletion
                            if f"confirm_delete_{book['id']}" not in st.session_state:
                                st.session_state[f"confirm_delete_{book['id']}"] = False
                            
                            if not st.session_state[f"confirm_delete_{book['id']}"]:
                                st.session_state[f"confirm_delete_{book['id']}"] = True
                                st.rerun()
                            else:
                                # Actually delete
                                if delete_book_from_supabase(book['id']):
                                    st.success(f"Book '{book.get('title', 'Untitled')}' deleted successfully!")
                                    st.session_state[f"confirm_delete_{book['id']}"] = False
                                    st.rerun()
                                else:
                                    st.session_state[f"confirm_delete_{book['id']}"] = False
                                    st.rerun()
                
                # Show confirmation message
                if st.session_state.get(f"confirm_delete_{book['id']}", False):
                    st.warning(f"⚠️ Click '🗑️ Delete' again to confirm deletion of '{book.get('title', 'Untitled')}'")
    


def render_book_iai_trees_page() -> None:
    """Render IAI Trees for a specific book"""
    book_id = st.session_state.get("selected_book_id")
    if not book_id:
        st.error("No book selected")
        return
    
    # Load book details
    client = get_supabase_client()
    if client is None:
        return
    
    try:
        book_resp = client.table("books").select("title, author").eq("id", book_id).execute()
        if not book_resp.data:
            st.error("Book not found")
            return
        book = book_resp.data[0]
    except Exception as e:
        st.error(f"Failed to load book: {e}")
        return
    
    # Header with Close button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"🌳 IAI Trees for: {book.get('title', 'Untitled')}")
    with col2:
        if st.button("❌ Close", use_container_width=True, type="secondary"):
            st.session_state.current_page = "library"
            st.rerun()
    
    if book.get('author'):
        st.write(f"**Author:** {book['author']}")
    
    # Load IAI Trees for this book
    iai_trees = load_iai_trees_for_book(book_id)
    
    if not iai_trees:
        st.info("No IAI Trees found for this book. Create an IAI Tree from the main page!")
        return
    
    st.subheader(f"📊 Found {len(iai_trees)} IAI Tree(s)")
    
    # Show IAI Trees
    for i, tree in enumerate(iai_trees):
        with st.expander(f"🌳 {tree.get('title', 'Untitled IAI Tree')} - {i+1}", expanded=True):
            # Show tree metadata
            created_at = tree.get('created_at', '')
            if created_at:
                date_str = created_at.split('T')[0] if 'T' in created_at else created_at[:10]
                time_str = created_at.split('T')[1][:5] if 'T' in created_at else created_at[:5]
                st.write(f"**Created:** {date_str} at {time_str}")
            
            # Render the IAI Tree
            html_content = tree.get('html_content', '')
            if html_content:
                st.subheader("🧠 IAI Framework Tree")
                components.html(html_content, height=680, scrolling=True)
            else:
                st.error("IAI Tree content not found")


def render_iai_tree_view_page() -> None:
    """Render saved IAI Tree view page"""
    tree = st.session_state.get("selected_iai_tree")
    if not tree:
        st.error("No IAI Tree selected")
        return
    
    # Header with Close button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"🌳 {tree.get('title', 'Untitled IAI Tree')}")
    with col2:
        if st.button("❌ Close", use_container_width=True, type="secondary"):
            st.session_state.current_page = "library"
            st.rerun()
    
    # Show tree metadata
    created_at = tree.get('created_at', '')
    if created_at:
        date_str = created_at.split('T')[0] if 'T' in created_at else created_at[:10]
        st.write(f"**Created:** {date_str}")
    
    # Get book title for context
    book_id = tree.get('book_id')
    if book_id:
        try:
            client = get_supabase_client()
            if client:
                book_resp = client.table("books").select("title").eq("id", book_id).execute()
                if book_resp.data:
                    book_title = book_resp.data[0].get('title', 'Unknown Book')
                    st.write(f"**Book:** {book_title}")
        except:
            pass
    
    st.divider()
    
    # Render the IAI Tree
    html_content = tree.get('html_content', '')
    if html_content:
        st.subheader("🧠 IAI Framework Tree")
        components.html(html_content, height=680, scrolling=True)
    else:
        st.error("IAI Tree content not found")


def render_book_detail_page() -> None:
    book_id = st.session_state.get("selected_book_id")
    if not book_id:
        st.error("No book selected")
        return
    
    # Load book details
    client = get_supabase_client()
    if client is None:
        return
    
    try:
        book_resp = client.table("books").select("*").eq("id", book_id).execute()
        if not book_resp.data:
            st.error("Book not found")
            return
        book = book_resp.data[0]
    except Exception as e:
        st.error(f"Failed to load book: {e}")
        return
    
    # Header with back button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Library"):
            st.session_state.current_page = "library"
            st.rerun()
    
    with col2:
        st.title(f"📖 {book.get('title', 'Untitled')}")
        if book.get("author"):
            st.write(f"**Author:** {book['author']}")
        if book.get("index_id"):
            st.write(f"**Index:** {book['index_id']}")
        if book.get("start_date"):
            st.write(f"**Started:** {book['start_date']}")
        if book.get("finish_date"):
            st.write(f"**Finished:** {book['finish_date']}")
    
    # Load and display summaries (saved content)
    summaries = load_summaries_for_book(book_id)
    
    if summaries:
        st.subheader("📝 Saved Content")
        
        # Group summaries by date
        from collections import defaultdict
        summaries_by_date = defaultdict(list)
        
        for summary in summaries:
            date_str = summary.get('created_at', 'Unknown date')[:10]  # YYYY-MM-DD
            summaries_by_date[date_str].append(summary)
        
        # Display grouped summaries
        for date_str in sorted(summaries_by_date.keys(), reverse=True):  # Most recent first
            date_summaries = summaries_by_date[date_str]
            with st.expander(f"📅 {date_str} ({len(date_summaries)} entries)", expanded=True):
                for i, summary in enumerate(date_summaries):
                    summary_id = summary['id']
                    content = summary.get("content", "")
                    
                    # Check if this entry is being edited
                    edit_key = f"edit_mode_{summary_id}"
                    if edit_key not in st.session_state:
                        st.session_state[edit_key] = False
                    
                    if st.session_state[edit_key]:
                        # Edit mode - show time instead of entry number
                        created_at = summary.get('created_at', '')
                        time_str = created_at.split('T')[1][:5] if 'T' in created_at else created_at[:5]
                        st.write(f"**{time_str} (Editing):**")
                        
                        # Text area for editing
                        edited_content = st.text_area(
                            "Edit content:",
                            value=content,
                            height=100,
                            key=f"edit_content_{summary_id}"
                        )
                        
                        # Edit buttons
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            if st.button("💾 Save", key=f"save_{summary_id}"):
                                if update_summary_in_supabase(summary_id, edited_content):
                                    st.success("Entry updated!")
                                    st.session_state[edit_key] = False
                                    st.rerun()
                        with col2:
                            if st.button("❌ Cancel", key=f"cancel_{summary_id}"):
                                st.session_state[edit_key] = False
                                st.rerun()
                        with col3:
                            if st.button("🗑️ Delete", key=f"delete_edit_{summary_id}"):
                                if delete_summary_from_supabase(summary_id):
                                    st.success("Entry deleted!")
                                    st.session_state[edit_key] = False
                                    st.rerun()
                    else:
                        # View mode
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            # Show time instead of entry number
                            created_at = summary.get('created_at', '')
                            time_str = created_at.split('T')[1][:5] if 'T' in created_at else created_at[:5]
                            st.write(f"**{time_str}:**")
                            if content.strip():
                                # Clean up any internal separators and display as bullet list
                                cleaned_content = content.replace("---", "").strip()
                                items = parse_lines_to_items(cleaned_content)
                                if items:
                                    for item in items:
                                        st.write(f"  • {item}")
                                else:
                                    st.write(f"  • {cleaned_content}")
                            else:
                                st.write("  (Empty content)")
                        
                        with col2:
                            # Create a nice button container
                            st.markdown("<div style='text-align: right; margin-top: 10px;'>", unsafe_allow_html=True)
                            
                            # Button row with better spacing
                            btn_col1, btn_col2 = st.columns([1, 1])
                            
                            with btn_col1:
                                if st.button("✏️", key=f"edit_{summary_id}", help="Edit this entry", use_container_width=True):
                                    st.session_state[edit_key] = True
                                    st.rerun()
                            
                            with btn_col2:
                                if st.button("🗑️", key=f"delete_{summary_id}", help="Delete this entry", use_container_width=True):
                                    if delete_summary_from_supabase(summary_id):
                                        st.success("Entry deleted!")
                                        st.rerun()
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    if i < len(date_summaries) - 1:  # Add separator between entries
                        st.write("---")
    else:
        st.info("No saved content yet. Go to the main page to add content and save it.")


def load_latest_iai_tree_for_book() -> None:
    """Load the latest IAI Tree for the current book"""
    if not st.session_state.get("book_id"):
        return
    
    try:
        client = get_supabase_client()
        if client is None:
            return
        
        # Get the latest IAI Tree for this book
        response = client.table("iai_trees").select("html_content").eq("book_id", st.session_state.book_id).eq("user_id", st.session_state.user.id).order("created_at", desc=True).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            st.session_state.mindmap_html = response.data[0]["html_content"]
    except Exception as e:
        # Silently fail to avoid disrupting the main flow
        pass


def render_main_page() -> None:
    # Load latest IAI Tree for current book if available
    if st.session_state.get("book_id") and not st.session_state.get("mindmap_html"):
        load_latest_iai_tree_for_book()
    
    settings = render_sidebar()
    render_input_ui()
    render_action_bar(settings)
    render_iai_tree()


def check_auth_session():
    """Check if user is already authenticated"""
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        # Check if there's an existing session
        session = supabase.auth.get_session()
        if session and session.user:
            st.session_state.user = session.user
            return True
    except:
        pass
    return False


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
    initialize_session_state()
    
    # Check for existing authentication session
    if st.session_state.user is None:
        check_auth_session()
    
    # Check authentication
    if st.session_state.user is None:
        render_auth_page()
        return
    
    # Initialize current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "main"
    
    # Navigation with logout
    if st.session_state.is_admin:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 2, 1])
        with col1:
            if st.button("🏠 Main"):
                # Clear all form data for fresh start
                st.session_state.book_title = ""
                st.session_state.book_author = ""
                st.session_state.book_start_date = None
                st.session_state.book_finish_date = None
                st.session_state.book_index_id = ""
                st.session_state.book_id = None
                st.session_state.original_book_title = ""
                st.session_state.input_text = ""
                # Clear IAI Tree from session state
                st.session_state.mindmap_html = None
                st.session_state.current_page = "main"
                st.rerun()
        with col2:
            if st.button("📚 Library"):
                st.session_state.current_page = "library"
                st.rerun()
        with col3:
            if st.button("🔧 Admin"):
                st.session_state.current_page = "admin"
                st.rerun()
        with col5:
            if st.button("🚪 Logout"):
                sign_out()
                st.rerun()
    else:
        col1, col2, col3, col4 = st.columns([1, 1, 3, 1])
        with col1:
            if st.button("🏠 Main"):
                # Clear all form data for fresh start
                st.session_state.book_title = ""
                st.session_state.book_author = ""
                st.session_state.book_start_date = None
                st.session_state.book_finish_date = None
                st.session_state.book_index_id = ""
                st.session_state.book_id = None
                st.session_state.original_book_title = ""
                st.session_state.input_text = ""
                # Clear IAI Tree from session state
                st.session_state.mindmap_html = None
                st.session_state.current_page = "main"
                st.rerun()
        with col2:
            if st.button("📚 Library"):
                st.session_state.current_page = "library"
                st.rerun()
        with col4:
            if st.button("🚪 Logout"):
                sign_out()
                st.rerun()
    
    # Show user info
    if st.session_state.is_admin:
        st.success(f"🔑 Admin mode: {st.session_state.user.email}")
        st.caption("🔧 Administrative access - view system statistics and manage the application")
    else:
        st.info(f"👤 Logged in as: {st.session_state.user.email}")
        st.caption("📚 Your personal book collection - only you can see your books and summaries")
    
    # Render current page
    if st.session_state.current_page == "library":
        render_library_page()
    elif st.session_state.current_page == "book_detail":
        render_book_detail_page()
    elif st.session_state.current_page == "book_iai_trees":
        render_book_iai_trees_page()
    elif st.session_state.current_page == "iai_tree_view":
        render_iai_tree_view_page()
    elif st.session_state.current_page == "admin":
        render_admin_dashboard()
    else:  # main page
        render_header()
        render_main_page()


if __name__ == "__main__":
    main()


