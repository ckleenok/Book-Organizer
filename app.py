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


def save_book_to_supabase() -> None:
    client = get_supabase_client()
    if client is None:
        return
    title = (st.session_state.book_title or "").strip()
    author = (st.session_state.book_author or "").strip()
    start_date_val = st.session_state.book_start_date
    finish_date_val = st.session_state.book_finish_date
    
    if not title:
        st.error("Title is required.")
        return
    
    # Check if this is a new book (different title than original)
    original_title = st.session_state.get("original_book_title", "")
    is_new_book = title != original_title
    
    if is_new_book:
        # Generate new index for new book, ensuring uniqueness
        today = date.today()
        base_date = start_date_val or today
        new_index_id = _generate_unique_index_id(base_date, client)
        st.session_state.book_index_id = new_index_id
        # Clear book_id to create new book
        st.session_state.book_id = None
    
    index_id = st.session_state.book_index_id or ""
    
    payload = {
        "title": title,
        "author": author or None,
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
    except Exception as e:
        st.error(f"Supabase error saving summary: {e}")


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
    """Generate a unique index ID by checking existing ones in the database"""
    month_key = f"{base_date.year}{base_date.month:02d}"
    seq = 1
    
    while True:
        index_id = _compute_index_id(base_date, seq)
        try:
            # Check if this index_id already exists
            existing = client.table("books").select("id").eq("index_id", index_id).execute()
            if not existing.data:
                # This index_id is available
                return index_id
            else:
                # This index_id exists, try next sequence
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
    if len(texts) == 0:
        return {"labels": [], "k": 0, "centers": None, "score": None}

    k = estimate_k(len(texts), grouping_strength, k_min, k_max)

    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        max_df=0.9,
        min_df=1,
        ngram_range=(1, 2),
    )
    pipeline = make_pipeline(vectorizer, Normalizer(copy=False))
    X = pipeline.fit_transform(texts)

    # Guard rails: K cannot exceed samples
    k = max(1, min(k, X.shape[0]))
    if k == 1:
        labels = np.zeros(X.shape[0], dtype=int)
        return {"labels": labels, "k": 1, "centers": None, "score": None}

    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(X)

    score = None
    try:
        if k > 1 and X.shape[0] > k:
            score = silhouette_score(X, labels)
    except Exception:
        score = None

    return {"labels": labels, "k": k, "centers": getattr(model, "cluster_centers_", None), "score": score}


def derive_cluster_names(texts: List[str], labels: np.ndarray, top_n: int = 3) -> Dict[int, str]:
    df = pd.DataFrame({"text": texts, "label": labels})
    names: Dict[int, str] = {}
    for label, group in df.groupby("label"):
        tokens: Dict[str, int] = {}
        for t in group["text"].tolist():
            for w in str(t).lower().replace("\n", " ").split():
                w = "".join([c for c in w if c.isalnum()])
                if len(w) <= 2:
                    continue
                tokens[w] = tokens.get(w, 0) + 1
        top = sorted(tokens.items(), key=lambda x: (-x[1], x[0]))[:top_n]
        if top:
            names[label] = ", ".join([w for w, _ in top])
        else:
            names[label] = f"Group {label + 1}"
    return names


def build_pyvis_mindmap(texts: List[str], labels: np.ndarray, cluster_names: Dict[int, str]) -> str:
    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=180, spring_strength=0.02)

    root_id = "root"
    net.add_node(root_id, label="Mind Map", shape="circle", color="#2f54eb")

    # Add cluster nodes
    cluster_to_node: Dict[int, str] = {}
    for label, name in cluster_names.items():
        node_id = f"cluster_{label}"
        cluster_to_node[label] = node_id
        net.add_node(node_id, label=name, shape="box", color="#69c0ff")
        net.add_edge(root_id, node_id, color="#91d5ff")

    # Add text nodes
    for idx, text in enumerate(texts):
        label = labels[idx]
        cluster_node = cluster_to_node[label]
        node_id = f"item_{idx}"
        numbered = f"{idx + 1}. {text}"
        preview = numbered if len(numbered) <= 140 else numbered[:137] + "..."
        net.add_node(node_id, label=preview, title=numbered, shape="dot", color="#ffd666")
        net.add_edge(cluster_node, node_id, color="#ffe58f")

    net.set_options(
        """
        {
          "nodes": {"scaling": {"min": 10, "max": 30}},
          "interaction": {"dragNodes": true, "dragView": true, "zoomView": true, "hover": true},
          "physics": {"stabilization": {"enabled": true, "iterations": 150}}
        }
        """
    )

    return net.generate_html(notebook=False)


def render_header() -> None:
    st.title(APP_TITLE)
    st.caption("Use Quick Add (Enter or Add button) to append items, or edit the list below. Then tap 'Mind Map Creation'.")


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Book")
    st.session_state.book_title = st.sidebar.text_input("Title", value=st.session_state.book_title)
    st.session_state.book_author = st.sidebar.text_input("Author", value=st.session_state.book_author)
    st.session_state.book_start_date = st.sidebar.date_input("Start date", value=st.session_state.book_start_date)
    st.session_state.book_finish_date = st.sidebar.date_input("Finish date", value=st.session_state.book_finish_date)

    # Auto Index (Year/Month/Sequence) based on input order
    st.sidebar.markdown("**Index (Year/Month/Sequence)**")
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

    st.sidebar.text_input("Generated Index", value=st.session_state.book_index_id, disabled=True)
    
    # Save book button
    if st.sidebar.button("Save Book", type="primary"):
        save_book_to_supabase()

    st.sidebar.header("Mind Map Settings")
    grouping = st.sidebar.slider("Grouping strength", 0.0, 1.0, 0.5, 0.05, help="Left: fewer, larger groups. Right: more, smaller groups.")
    k_min = st.sidebar.number_input("Min groups", min_value=1, max_value=50, value=2, step=1)
    k_max = st.sidebar.number_input("Max groups", min_value=1, max_value=100, value=8, step=1)
    return {"grouping_strength": grouping, "k_min": int(k_min), "k_max": int(k_max)}


def render_input_ui() -> None:
    st.subheader("Input")
    # Quick add: press Enter to save individual item immediately
    def _on_quick_add() -> None:
        value = st.session_state.get("quick_add", "").strip()
        if value and st.session_state.get("book_id"):
            # Clean up pasted content - remove everything after '-<' or '- <'
            cleaned_value = value
            if '-<' in cleaned_value:
                cleaned_value = cleaned_value.split('-<')[0].strip()
            elif '- <' in cleaned_value:
                cleaned_value = cleaned_value.split('- <')[0].strip()
            
            if cleaned_value:
                # Save immediately as individual entry
                save_summary_to_supabase(cleaned_value)
            
            st.session_state.quick_add = ""
        elif value and not st.session_state.get("book_id"):
            st.warning("Please save the book first before adding content.")
            st.session_state.quick_add = ""

    st.text_input(
        label="Quick Add (press Enter)",
        key="quick_add",
        placeholder="Type an item and press Enter",
        on_change=_on_quick_add,
    )
    st.button("Add Item", on_click=_on_quick_add, type="secondary", use_container_width=True)

    # Show recent entries for this book
    if st.session_state.get("book_id"):
        summaries = load_summaries_for_book(st.session_state.book_id)
        if summaries:
            st.caption("Recent entries")
            for i, summary in enumerate(summaries[-3:]):  # Show last 3 entries
                content = summary.get("content", "").strip()
                if content:
                    st.write(f"• {content}")


# Removed multi-row helpers in favor of single textarea input


def render_action_bar(settings: Dict[str, Any]) -> None:
    left, right = st.columns([1, 1])
    with left:
        if st.button("Mind Map Creation", type="primary"):
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
                    html = build_pyvis_mindmap(texts, np.array(result["labels"]), names)
                    st.session_state.mindmap_html = html
    with right:
        if st.session_state.mindmap_html:
            st.download_button(
                label="Download HTML",
                data=st.session_state.mindmap_html,
                file_name="mind_map.html",
                mime="text/html",
            )


def render_mindmap() -> None:
    if st.session_state.mindmap_html:
        st.subheader("Mind Map")
        components.html(st.session_state.mindmap_html, height=680, scrolling=True)


def load_books_from_supabase() -> List[Dict[str, Any]]:
    client = get_supabase_client()
    if client is None:
        return []
    try:
        resp = client.table("books").select("*").order("created_at", desc=True).execute()
        return resp.data or []
    except Exception as e:
        st.error(f"Failed to load books: {e}")
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
        return True
    except Exception as e:
        st.error(f"Failed to delete summary: {e}")
        return False


def render_library_page() -> None:
    st.title("📚 Book Library")
    
    books = load_books_from_supabase()
    
    if not books:
        st.info("No books found. Go to the main page to add your first book!")
        return
    
    # Filters
    st.subheader("🔍 Filters")
    
    # All filters in one row
    col1, col2, col3 = st.columns([3, 1, 1])
    
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
    
    # Book cards
    for book in books:
        # Get entry count for this book
        summaries = load_summaries_for_book(book['id'])
        entry_count = len(summaries)
        
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.subheader(book.get("title", "Untitled"))
                if book.get("author"):
                    st.write(f"**Author:** {book['author']}")
                if book.get("index_id"):
                    st.write(f"**Index:** {book['index_id']}")
                if book.get("start_date"):
                    st.write(f"**Started:** {book['start_date']}")
                if book.get("finish_date") and book.get("finish_date") != "None":
                    st.write(f"**Finished:** {book['finish_date']}")
                # Show entry count
                if entry_count > 0:
                    st.write(f"📝 **{entry_count} entries**")
                else:
                    st.write("📝 **No entries yet**")
            
            with col2:
                if st.button(f"📖 View", key=f"view_{book['id']}"):
                    st.session_state.selected_book_id = book['id']
                    st.session_state.current_page = "book_detail"
                    st.rerun()
            
            with col3:
                if st.button(f"✏️ Edit", key=f"edit_{book['id']}"):
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
            
            with col4:
                if st.button(f"🗑️ Delete", key=f"delete_{book['id']}", type="secondary"):
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
            
            st.divider()


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
            with st.expander(f"📅 {date_str} ({len(date_summaries)} entries)"):
                for i, summary in enumerate(date_summaries):
                    # Create columns for content and delete button
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.write(f"**Entry {i+1}:**")
                        content = summary.get("content", "")
                        if content.strip():
                            # Clean up any internal separators and display as numbered list
                            cleaned_content = content.replace("---", "").strip()
                            items = parse_lines_to_items(cleaned_content)
                            if items:
                                for j, item in enumerate(items, 1):
                                    st.write(f"  {j}. {item}")
                            else:
                                st.write(f"  {cleaned_content}")
                        else:
                            st.write("  (Empty content)")
                    
                    with col2:
                        st.write("")  # Add some spacing
                        st.write("")  # Add some spacing
                        if st.button("🗑️", key=f"delete_summary_{summary['id']}", help="Delete this entry"):
                            if delete_summary_from_supabase(summary['id']):
                                st.success("Entry deleted!")
                                st.rerun()
                    
                    if i < len(date_summaries) - 1:  # Add separator between entries
                        st.write("---")
    else:
        st.info("No saved content yet. Go to the main page to add content and save it.")


def render_main_page() -> None:
    settings = render_sidebar()
    render_input_ui()
    render_action_bar(settings)
    render_mindmap()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
    initialize_session_state()
    
    # Initialize current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "main"
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 4])
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
            st.session_state.current_page = "main"
            st.rerun()
    with col2:
        if st.button("📚 Library"):
            st.session_state.current_page = "library"
            st.rerun()
    
    # Render current page
    if st.session_state.current_page == "library":
        render_library_page()
    elif st.session_state.current_page == "book_detail":
        render_book_detail_page()
    else:  # main page
        render_header()
        render_main_page()


if __name__ == "__main__":
    main()


