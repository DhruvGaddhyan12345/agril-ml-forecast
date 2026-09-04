FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[full]"
RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
