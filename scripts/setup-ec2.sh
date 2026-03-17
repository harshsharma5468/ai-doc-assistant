#!/usr/bin/env bash
# =============================================================================
# AWS EC2 Production Setup Script
# Tested on Amazon Linux 2023 / Ubuntu 22.04
# Run as root or with sudo
# =============================================================================

set -euo pipefail

APP_DIR="/opt/ai-doc-assistant"
COMPOSE_VERSION="2.27.0"
LOG_FILE="/var/log/aidoc-setup.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
die() { log "ERROR: $*"; exit 1; }

# ── Detect OS ─────────────────────────────────────────────────────────────────
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    die "Cannot detect OS"
fi
log "Detected OS: $OS"

# ── System Update & Packages ───────────────────────────────────────────────────
log "Updating system packages..."
if [[ "$OS" == "amzn" ]]; then
    yum update -y
    yum install -y git curl wget htop unzip jq
elif [[ "$OS" == "ubuntu" ]]; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
    apt-get install -y git curl wget htop unzip jq
fi

# ── Docker ────────────────────────────────────────────────────────────────────
log "Installing Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | bash
    systemctl enable docker
    systemctl start docker

    # Add current user to docker group
    SUDO_USER="${SUDO_USER:-ec2-user}"
    usermod -aG docker "$SUDO_USER" || true
fi

# ── Docker Compose v2 ─────────────────────────────────────────────────────────
log "Installing Docker Compose v${COMPOSE_VERSION}..."
if ! docker compose version &>/dev/null; then
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -SL "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi
docker compose version

# ── App Directory ─────────────────────────────────────────────────────────────
log "Setting up app directory: $APP_DIR"
mkdir -p "$APP_DIR"/{data/{uploads,vectorstore},logs,backups}
chmod 755 "$APP_DIR"

# ── Clone or Update Repo ──────────────────────────────────────────────────────
if [ -d "$APP_DIR/.git" ]; then
    log "Updating existing repository..."
    cd "$APP_DIR" && git pull --rebase origin main
else
    log "Cloning repository..."
    # Replace with your actual repo URL
    git clone https://github.com/YOUR_ORG/ai-doc-assistant.git "$APP_DIR"
fi

# ── Environment File ──────────────────────────────────────────────────────────
log "Configuring environment..."
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<EOF
# ── LLM Provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# ── Vector Store ───────────────────────────────────────────────────────────────
VECTOR_STORE_TYPE=faiss
VECTOR_STORE_PATH=/app/data/vectorstore
UPLOAD_DIR=/app/data/uploads

# ── App ────────────────────────────────────────────────────────────────────────
ENVIRONMENT=production
SECRET_KEY=$(openssl rand -hex 32)
LOG_LEVEL=INFO

# ── Redis ──────────────────────────────────────────────────────────────────────
REDIS_URL=redis://redis:6379/0

# ── Monitoring ─────────────────────────────────────────────────────────────────
GRAFANA_PASSWORD=$(openssl rand -hex 16)
METRICS_ENABLED=true
ENABLE_TRACING=true
EOF
    log "Created .env — EDIT IT with your API keys: $ENV_FILE"
    echo ""
    echo "⚠️  IMPORTANT: Edit $ENV_FILE and set OPENAI_API_KEY before starting!"
    echo ""
fi

# ── Systemd Service ───────────────────────────────────────────────────────────
log "Installing systemd service..."
cat > /etc/systemd/system/ai-doc-assistant.service <<EOF
[Unit]
Description=AI Document Assistant
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/local/bin/docker compose -f docker/docker-compose.yml up -d --remove-orphans
ExecStop=/usr/local/bin/docker compose -f docker/docker-compose.yml down
ExecReload=/usr/local/bin/docker compose -f docker/docker-compose.yml pull && \
           /usr/local/bin/docker compose -f docker/docker-compose.yml up -d --remove-orphans
TimeoutStartSec=300
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Create symlink so compose is in PATH
ln -sf /usr/local/lib/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose || true

systemctl daemon-reload
systemctl enable ai-doc-assistant

# ── Logrotate ─────────────────────────────────────────────────────────────────
log "Configuring log rotation..."
cat > /etc/logrotate.d/ai-doc-assistant <<EOF
$APP_DIR/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}
EOF

# ── Firewall ──────────────────────────────────────────────────────────────────
log "Configuring firewall..."
if command -v ufw &>/dev/null; then
    ufw allow 22/tcp
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw --force enable
fi

# ── Backup Script ─────────────────────────────────────────────────────────────
cat > /usr/local/bin/aidoc-backup.sh <<'BACKUP_EOF'
#!/bin/bash
set -euo pipefail
BACKUP_DIR="/opt/ai-doc-assistant/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"

# Backup vector store and uploads
tar czf "$BACKUP_FILE" \
    -C /opt/ai-doc-assistant/data \
    vectorstore uploads 2>/dev/null || true

# Keep last 7 days of backups
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +7 -delete

echo "Backup created: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
BACKUP_EOF
chmod +x /usr/local/bin/aidoc-backup.sh

# Daily backup cron
(crontab -l 2>/dev/null || true; echo "0 2 * * * /usr/local/bin/aidoc-backup.sh >> /var/log/aidoc-backup.log 2>&1") | crontab -

# ── Health Check Script ───────────────────────────────────────────────────────
cat > /usr/local/bin/aidoc-health.sh <<'HEALTH_EOF'
#!/bin/bash
BASE_URL="${1:-http://localhost:8000}"
echo "=== AI Doc Assistant Health Check ==="
curl -s "$BASE_URL/health/" | python3 -m json.tool 2>/dev/null || echo "API unreachable"
echo ""
echo "=== Docker Service Status ==="
docker compose -f /opt/ai-doc-assistant/docker/docker-compose.yml ps
HEALTH_EOF
chmod +x /usr/local/bin/aidoc-health.sh

# ── Summary ───────────────────────────────────────────────────────────────────
log "Setup complete!"
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          AI Document Assistant — Setup Complete          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  1. Edit API keys:   $ENV_FILE   ║"
echo "║  2. Start services:  systemctl start ai-doc-assistant    ║"
echo "║  3. Check status:    aidoc-health.sh                     ║"
echo "║                                                          ║"
echo "║  URLs (after start):                                     ║"
echo "║  • UI:          http://YOUR_EC2_IP                       ║"
echo "║  • API:         http://YOUR_EC2_IP/api/v1                ║"
echo "║  • API Docs:    http://YOUR_EC2_IP/docs                  ║"
echo "║  • Grafana:     http://YOUR_EC2_IP:3000                  ║"
echo "║  • Prometheus:  http://YOUR_EC2_IP:9090                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
