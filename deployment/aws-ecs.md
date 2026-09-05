# AWS ECS Deployment

1. Create an ECR repository for the API image.
2. Create an RDS PostgreSQL instance.
3. Build and push the image:

```bash
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
docker build -t fake-news-api .
docker tag fake-news-api:latest ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/fake-news-api:latest
docker push ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/fake-news-api:latest
```

4. Create an ECS Fargate task definition using the image.
5. Set `DATABASE_URL` to the RDS PostgreSQL connection string.
6. Attach an Application Load Balancer and health check `/health`.
