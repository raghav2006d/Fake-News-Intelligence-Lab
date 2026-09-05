# Render Deployment

1. Push this repo to GitHub.
2. In Render, create a Blueprint from `render.yaml`.
3. Render will create:
   - `fake-news-api`
   - `fake-news-postgres`
4. Set frontend `VITE_API_URL` to the Render API URL before deploying the React app.

API health check:

```text
https://YOUR-RENDER-SERVICE.onrender.com/health
```
