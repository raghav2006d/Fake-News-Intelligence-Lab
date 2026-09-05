# Netlify Frontend Deployment

1. Push this repo to GitHub.
2. Create a Netlify site from the `frontend` directory.
3. Netlify will use `frontend/netlify.toml`.
4. Add this environment variable:

```text
VITE_API_URL=https://YOUR-API-DOMAIN
```

5. Deploy the site.
