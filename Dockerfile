# Use Red Hat UBI 9 Python as the base image
FROM registry.access.redhat.com/ubi9/python-312:latest

USER root

# Optional: Apply security updates only
RUN dnf -y upgrade --refresh && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Set working directory
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY ./app /app/app

# OpenShift-friendly permissions:
# - OpenShift runs with an arbitrary UID, usually in GID 0
# - Make files group-owned by 0 and group-readable/writable to match user perms
RUN chgrp -R 0 /app && chmod -R g=u /app

# Drop privileges (OpenShift will override UID anyway)
USER 1001

# Expose port used by OpenShift routes/services
EXPOSE 8080

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]