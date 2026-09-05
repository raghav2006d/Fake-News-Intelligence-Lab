# Railway API Deployment

1. Push this repo to GitHub.
2. Create a Railway project from the repository root.
3. Railway will use `railway.json` and the root `Dockerfile`.
4. Add a PostgreSQL database plugin in Railway.
5. Set this environment variable for the API service:

```text
DATABASE_URL=${{Postgres.DATABASE_URL}}
```

6. Use the generated Railway domain as the frontend `VITE_API_URL`.
