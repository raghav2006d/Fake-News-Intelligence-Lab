# AWS EC2 Deployment

1. Create an Ubuntu EC2 instance and open ports `80`, `443`, and `8000`.
2. Install Docker and Docker Compose.
3. Clone this repository on the instance.
4. Create a `.env` file with production values:

```bash
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DB
```

5. Start the API:

```bash
docker compose up --build -d api postgres
```

6. Put Nginx in front of `localhost:8000` and enable HTTPS with Certbot.

For a stronger version, push the API image to Amazon ECR and run it on ECS Fargate with RDS PostgreSQL.
