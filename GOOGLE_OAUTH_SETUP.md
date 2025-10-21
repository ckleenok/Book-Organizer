# Google OAuth Setup Instructions

To enable Gmail/Google sign-in in your Book Organizer app, follow these steps:

## 1. Google Cloud Console Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google+ API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google+ API" and enable it
4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - Choose "Web application"
   - Add authorized redirect URIs:
     - `https://your-supabase-project.supabase.co/auth/v1/callback`
     - `http://localhost:8501/` (for local development)

## 2. Supabase Dashboard Setup

1. Go to your Supabase project dashboard
2. Navigate to "Authentication" > "Providers"
3. Enable Google provider
4. Add your Google OAuth credentials:
   - **Client ID**: From Google Cloud Console
   - **Client Secret**: From Google Cloud Console
5. Set redirect URL: `https://your-supabase-project.supabase.co/auth/v1/callback`

## 3. Update Supabase URL (if needed)

Make sure your `supabase_url` in the app matches your project URL:
- Format: `https://your-project-id.supabase.co`

## 4. Test the Integration

1. Run your Streamlit app
2. Click "Sign in with Google"
3. Complete the OAuth flow
4. You should be redirected back to your app

## Troubleshooting

- **Redirect URI mismatch**: Make sure the redirect URI in Google Console matches your Supabase callback URL
- **Invalid client**: Double-check your Client ID and Secret in Supabase
- **CORS issues**: Ensure your domain is added to authorized origins in Google Console

## Security Notes

- Keep your Client Secret secure
- Use HTTPS in production
- Regularly rotate your OAuth credentials
- Monitor authentication logs in Supabase
