# 데이터베이스 업데이트 가이드

현재 오류: `column books.user_id does not exist`

## 해결 방법:

### 1. Supabase Dashboard에서 SQL Editor 열기
1. [Supabase Dashboard](https://supabase.com/dashboard) 접속
2. 프로젝트 선택
3. 왼쪽 메뉴에서 **"SQL Editor"** 클릭

### 2. 다음 SQL 명령어 실행

```sql
-- books 테이블에 user_id 컬럼 추가 (이미 존재하면 오류 무시)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'books' AND column_name = 'user_id') THEN
        ALTER TABLE books ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
    END IF;
END $$;

-- books 테이블에 isbn 컬럼 추가 (이미 존재하면 오류 무시)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'books' AND column_name = 'isbn') THEN
        ALTER TABLE books ADD COLUMN isbn TEXT;
    END IF;
END $$;

-- 기존 데이터가 있다면 NULL로 설정 (선택사항)
-- UPDATE books SET isbn = NULL WHERE isbn IS NULL;

-- summaries 테이블은 이미 book_id로 연결되어 있으므로 수정 불필요

-- 인덱스 추가 (성능 향상)
CREATE INDEX IF NOT EXISTS idx_books_user_id ON books(user_id);

-- Row Level Security 활성화
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

### 3. 기존 데이터 처리 (선택사항)
만약 기존에 데이터가 있다면, 다음 명령어로 기본 사용자에게 할당:

```sql
-- 기존 books의 user_id를 첫 번째 사용자로 설정 (필요한 경우)
-- UPDATE books SET user_id = (SELECT id FROM auth.users LIMIT 1) WHERE user_id IS NULL;
```

### 4. 확인
SQL 실행 후 앱을 새로고침하면 오류가 해결됩니다.

## 5. 오류 해결

### 현재 오류: `Could not find the 'isbn' column of 'books' in the schema cache`

**해결 방법:**
1. Supabase Dashboard > SQL Editor에서 위의 SQL 실행
2. 특히 다음 명령어가 중요:
   ```sql
   ALTER TABLE books ADD COLUMN isbn TEXT;
   ```
3. 앱을 새로고침하고 다시 시도

### 오류: `column "user_id" of relation "books" already exists`

**해결 방법:**
- `user_id` 컬럼이 이미 존재하므로 무시하고 `isbn` 컬럼만 추가:
  ```sql
  ALTER TABLE books ADD COLUMN isbn TEXT;
  ```
- 또는 위의 안전한 SQL을 사용하여 오류 없이 실행

### 추가 확인사항:
- `user_id` 컬럼도 추가되었는지 확인
- RLS 정책이 활성화되었는지 확인
- 기존 데이터가 있다면 백업 권장

## 주의사항
- 이 작업은 기존 데이터에 영향을 줄 수 있습니다
- 프로덕션 환경에서는 백업을 먼저 수행하세요
- RLS 정책이 활성화되면 사용자는 자신의 데이터만 볼 수 있습니다
