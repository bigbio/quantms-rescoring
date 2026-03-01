# ===================
# Stage 1: Build
# ===================
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    libhdf5-dev \
    libnetcdf-dev \
    libxml2 \
    libxslt1.1 \
    libssl3 \
    libffi-dev \
    procps \
 && rm -rf /var/lib/apt/lists/*

#RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
#    locale-gen

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir pip==25.1.1 && \
    pip install --no-cache-dir poetry==2.1.3 && \
    poetry build && \
    pip install --no-cache-dir dist/*.whl && \
    pip cache purge

# ===================
# Stage 2: Runtime
# ===================
FROM python:3.11-slim AS final

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    libhdf5-dev \
    libnetcdf-dev \
    libxml2 \
    libxslt1.1 \
    libssl3 \
    libffi-dev \
    procps \
 && rm -rf /var/lib/apt/lists/*

#RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
#    locale-gen

COPY --from=builder /usr/local /usr/local

ENV HOME=/app
ENV PEPTDEEP_HOME=/app
ENV MPLCONFIGDIR=/app/.config/matplotlib

WORKDIR /app
RUN chmod -R 755 /app

RUN python3.11 -c "import pyopenms; print('pyOpenMS imported successfully')"