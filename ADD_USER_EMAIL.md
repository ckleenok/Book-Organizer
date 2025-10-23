# Add User Email to Books Table

## Database Update Required

To show email addresses in the admin dashboard, we need to add a `user_email` field to the books table.

### 1. Supabase Dashboard Access
1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Click on **"SQL Editor"** in the left menu

### 2. Execute the following SQL:

```sql
-- Add user_email column to books table
ALTER TABLE books ADD COLUMN user_email TEXT;

-- Create index for better performance
CREATE INDEX IF NOT EXISTS idx_books_user_email ON books(user_email);

-- Update existing books with user emails (if any exist)
-- Note: This will only work for books created after this update
-- Existing books will have NULL user_email until they are updated
```

### 3. Alternative: Manual Update (if needed)

If you want to update existing books with user emails, you can run:

```sql
-- This is optional - only run if you have existing books and want to update them
-- You'll need to manually set the emails for existing books
UPDATE books SET user_email = 'user@example.com' WHERE user_id = 'user-uuid-here';
```

### 4. Verification

After running the SQL, you can verify the column was added:

```sql
-- Check if the column exists
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'books' AND column_name = 'user_email';
```

## What This Enables

- **Email Display**: Admin dashboard will show user email addresses instead of user IDs
- **Better User Identification**: Easier to identify users in the admin interface
- **Privacy Compliant**: Only stores emails that users have already provided during registration

## Important Notes

- **New Books Only**: The `user_email` field will only be populated for books created after this database update
- **Existing Books**: Books created before this update will show "Unknown email" until they are updated
- **No Data Loss**: This is a non-destructive change that only adds a new column
