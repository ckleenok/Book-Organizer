# Book Organizer - Mind Map App

A Streamlit web application for organizing books and creating mind maps from your reading notes.

## Features

- **Book Management**: Add books with metadata (title, author, start/finish dates, auto-generated index)
- **Individual Entry System**: Add reading notes as separate entries
- **Mind Map Generation**: Create interactive mind maps from your notes using AI clustering
- **Library View**: Browse, search, and filter your book collection
- **Supabase Integration**: Persistent storage for all your data

## How to Use

1. **Add a Book**: Fill in book details and click "Save Book"
2. **Add Notes**: Use "Quick Add" to add individual reading notes
3. **Create Mind Maps**: Click "Mind Map Creation" to generate interactive visualizations
4. **Browse Library**: Use filters to find books by year, month, or search terms

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run locally: `streamlit run app.py`
3. For deployment, configure your Supabase credentials

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Supabase (PostgreSQL)
- **AI/ML**: scikit-learn for clustering
- **Visualization**: Pyvis for interactive mind maps
