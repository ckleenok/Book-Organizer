# 데이터베이스 완전 초기화 가이드

기존 테이블과 데이터를 모두 삭제하고 새로 생성합니다.

## 1. Supabase Dashboard 접속
1. [Supabase Dashboard](https://supabase.com/dashboard) 접속
2. 프로젝트 선택
3. 왼쪽 메뉴에서 **"SQL Editor"** 클릭

## 2. 기존 테이블 및 데이터 삭제

```sql
-- 기존 테이블 삭제 (순서 중요: 외래키 때문에)
DROP TABLE IF EXISTS summaries CASCADE;
DROP TABLE IF EXISTS books CASCADE;

-- 기존 정책 삭제 (있다면)
DROP POLICY IF EXISTS "Users can view their own books" ON books;
DROP POLICY IF EXISTS "Users can insert their own books" ON books;
DROP POLICY IF EXISTS "Users can update their own books" ON books;
DROP POLICY IF EXISTS "Users can delete their own books" ON books;

DROP POLICY IF EXISTS "Users can view summaries of their books" ON summaries;
DROP POLICY IF EXISTS "Users can insert summaries to their books" ON summaries;
DROP POLICY IF EXISTS "Users can update summaries of their books" ON summaries;
DROP POLICY IF EXISTS "Users can delete summaries of their books" ON summaries;

-- 기존 인덱스 삭제 (있다면)
DROP INDEX IF EXISTS idx_books_user_id;
DROP INDEX IF EXISTS idx_books_index_id;
DROP INDEX IF EXISTS idx_summaries_book_id;
```

## 3. 새 테이블 생성

```sql
-- books 테이블 생성 (완전한 스키마)
CREATE TABLE books (
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

-- summaries 테이블 생성
CREATE TABLE summaries (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    is_manual BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 인덱스 생성
CREATE INDEX idx_books_user_id ON books(user_id);
CREATE INDEX idx_books_index_id ON books(index_id);
CREATE INDEX idx_summaries_book_id ON summaries(book_id);
```

## 4. RLS (Row Level Security) 설정

```sql
-- RLS 활성화
ALTER TABLE books ENABLE ROW LEVEL SECURITY;
ALTER TABLE summaries ENABLE ROW LEVEL SECURITY;

-- RLS 정책 생성
CREATE POLICY "Users can view their own books" ON books FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own books" ON books FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own books" ON books FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own books" ON books FOR DELETE USING (auth.uid() = user_id);

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
```

## 5. 확인

```sql
-- 테이블 구조 확인
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name IN ('books', 'summaries')
ORDER BY table_name, ordinal_position;

-- RLS 정책 확인
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual
FROM pg_policies 
WHERE tablename IN ('books', 'summaries');
```

## 6. 앱 테스트

1. 앱을 새로고침
2. 로그인/회원가입
3. 책 등록 테스트
4. ISBN 검색 테스트

## 주의사항

⚠️ **이 작업은 모든 기존 데이터를 삭제합니다!**
- 백업이 필요한 데이터가 있다면 먼저 백업하세요
- 프로덕션 환경에서는 신중하게 진행하세요
- 모든 사용자 데이터가 삭제됩니다
