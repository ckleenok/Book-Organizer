# IAI Tree RLS Policy Fix

## 문제
IAI Tree 저장 시 RLS 정책 위반 오류 발생:
```
Supabase error saving IAI Tree: {'message': 'new row violates row-level security policy for table "iai_trees"', 'code': '42501'}
```

## 해결 방법

### 1. Supabase SQL Editor에서 다음 SQL 실행:

```sql
-- 먼저 기존 정책 삭제 (있다면)
DROP POLICY IF EXISTS "Users can view their own iai_trees" ON iai_trees;
DROP POLICY IF EXISTS "Users can insert their own iai_trees" ON iai_trees;
DROP POLICY IF EXISTS "Users can update their own iai_trees" ON iai_trees;
DROP POLICY IF EXISTS "Users can delete their own iai_trees" ON iai_trees;

-- 새로운 정책 생성
CREATE POLICY "Users can view their own iai_trees" ON iai_trees 
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own iai_trees" ON iai_trees 
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own iai_trees" ON iai_trees 
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own iai_trees" ON iai_trees 
    FOR DELETE USING (auth.uid() = user_id);
```

### 2. 임시로 RLS 비활성화 (테스트용)

만약 위 방법이 작동하지 않으면, 임시로 RLS를 비활성화해서 테스트:

```sql
-- 임시로 RLS 비활성화 (주의: 보안상 프로덕션에서는 권장하지 않음)
ALTER TABLE iai_trees DISABLE ROW LEVEL SECURITY;
```

### 3. 사용자 인증 확인

앱에서 사용자 인증이 제대로 되고 있는지 확인:
- 로그인 상태 확인
- `st.session_state.user.id` 값이 올바른지 확인

## 디버깅

### 현재 사용자 ID 확인:
```sql
SELECT auth.uid() as current_user_id;
```

### 테이블 구조 확인:
```sql
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'iai_trees';
```

### RLS 정책 확인:
```sql
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check
FROM pg_policies 
WHERE tablename = 'iai_trees';
```
