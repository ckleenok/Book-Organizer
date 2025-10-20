-- Create books table
CREATE TABLE IF NOT EXISTS books (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    start_date DATE,
    finish_date DATE,
    index_id TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create summaries table
CREATE TABLE IF NOT EXISTS summaries (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    is_manual BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_books_index_id ON books(index_id);
CREATE INDEX IF NOT EXISTS idx_summaries_book_id ON summaries(book_id);

-- Enable Row Level Security (RLS) - optional, adjust based on your auth needs
-- ALTER TABLE books ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE summaries ENABLE ROW LEVEL SECURITY;

-- Example RLS policies (uncomment if you want to enable RLS)
-- CREATE POLICY "Users can view their own books" ON books FOR SELECT USING (auth.uid() = user_id);
-- CREATE POLICY "Users can insert their own books" ON books FOR INSERT WITH CHECK (auth.uid() = user_id);
-- CREATE POLICY "Users can update their own books" ON books FOR UPDATE USING (auth.uid() = user_id);
-- CREATE POLICY "Users can delete their own books" ON books FOR DELETE USING (auth.uid() = user_id);

-- Similar policies for summaries table
-- CREATE POLICY "Users can view summaries of their books" ON summaries FOR SELECT USING (
--     book_id IN (SELECT id FROM books WHERE auth.uid() = user_id)
-- );
-- CREATE POLICY "Users can insert summaries to their books" ON summaries FOR INSERT WITH CHECK (
--     book_id IN (SELECT id FROM books WHERE auth.uid() = user_id)
-- );
