FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pypdf

COPY parser_service/ /app/parser_service/

EXPOSE 8001

CMD ["uvicorn", "parser_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
