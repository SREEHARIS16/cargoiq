FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Generate data, build features, and train models at image build time
# so the container is self-contained and works with zero external setup.
RUN python src/data_generation.py \
    && python src/preprocessing.py \
    && cd src && python train.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
