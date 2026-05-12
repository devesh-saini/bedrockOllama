FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV DJANGO_SECRET_KEY=""
ENV REGION=""
ENV BEDROCK_KB_ID=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV SENTRY_DSN=""
ENV ALLOWED_HOST=""

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python DockerRag/manage.py collectstatic --noinput

EXPOSE 8000

CMD ["sh", "-c", "cd DockerRag && gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 DockerRag.wsgi:application"]