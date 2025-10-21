# 데이터베이스 업데이트 가이드

현재 오류: `column books.user_id does not exist`

## 해결 방법:

### 1. Supabase Dashboard에서 SQL Editor 열기
1. [Supabase Dashboard](https://supabase.com/dashboard) 접속
2. 프로젝트 선택
3. 왼쪽 메뉴에서 **"SQL Editor"** 클릭

### 2. 다음 SQL 명령어 실행

```sql
-- books 테이블에 user_id 컬럼 추가
ALTER TABLE books ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

-- books 테이블에 isbn 컬럼 추가
ALTER TABLE books ADD COLUMN isbn TEXT;

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

## 주의사항
- 이 작업은 기존 데이터에 영향을 줄 수 있습니다
- 프로덕션 환경에서는 백업을 먼저 수행하세요
- RLS 정책이 활성화되면 사용자는 자신의 데이터만 볼 수 있습니다
