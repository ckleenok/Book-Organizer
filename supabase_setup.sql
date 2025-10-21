-- Create books table
CREATE TABLE IF NOT EXISTS books (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    author TEXT,
    isbn TEXT,
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

-- Create iai_trees table
CREATE TABLE IF NOT EXISTS iai_trees (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    html_content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_books_index_id ON books(index_id);
CREATE INDEX IF NOT EXISTS idx_summaries_book_id ON summaries(book_id);
CREATE INDEX IF NOT EXISTS idx_iai_trees_book_id ON iai_trees(book_id);
CREATE INDEX IF NOT EXISTS idx_iai_trees_user_id ON iai_trees(user_id);

-- Enable Row Level Security (RLS)
ALTER TABLE books ENABLE ROW LEVEL SECURITY;
ALTER TABLE summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE iai_trees ENABLE ROW LEVEL SECURITY;

-- RLS policies for books table
CREATE POLICY "Users can view their own books" ON books FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own books" ON books FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own books" ON books FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own books" ON books FOR DELETE USING (auth.uid() = user_id);

-- RLS policies for summaries table
CREATE POLICY "Users can view summaries of their books" ON summaries FOR SELECT USING (
    book_id IN (SELECT id FROM books WHERE auth.uid() = user_id)
);
CREATE POLICY "Users can insert summaries to their books" ON summaries FOR INSERT WITH CHECK (
    book_id IN (SELECT id FROM books WHERE auth.uid() = user_id)
);
CREATE POLICY "Users can update summaries of their books" ON summaries FOR UPDATE USING (
    book_id IN (SELECT id FROM books WHERE auth.uid() = user_id)
);
CREATE POLICY "Users can delete summaries of their books" ON summaries FOR DELETE USING (
    book_id IN (SELECT id FROM books WHERE auth.uid() = user_id)
);

-- RLS policies for iai_trees table
CREATE POLICY "Users can view their own iai_trees" ON iai_trees FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own iai_trees" ON iai_trees FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own iai_trees" ON iai_trees FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own iai_trees" ON iai_trees FOR DELETE USING (auth.uid() = user_id);
