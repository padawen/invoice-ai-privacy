#!/bin/bash

# Oracle Cloud Free Tier Setup Script for Invoice AI Privacy
# Run this script on a fresh Oracle Cloud VM (Ubuntu 22.04 ARM64)

set -e

echo "=== Invoice AI Privacy - Oracle Cloud Setup ==="
echo "Setting up privacy-focused invoice processing on Oracle Cloud Free Tier"
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
echo "📦 Installing Docker Compose..."
DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git
echo "📦 Installing Git..."
sudo apt-get install -y git

# Clone the repository
echo "📂 Cloning Invoice AI Privacy repository..."
cd ~
if [ -d "invoice-ai-privacy" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd invoice-ai-privacy
    git pull
else
    git clone https://github.com/your-username/invoice-ai-privacy.git
    cd invoice-ai-privacy
fi

# Create environment file
echo "⚙️ Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file - please customize if needed"
else
    echo ".env file already exists"
fi

# Configure for Oracle Cloud (ARM64)
echo "🔧 Configuring for Oracle Cloud ARM64..."
echo "OLLAMA_MODEL=phi3:mini" >> .env
echo "HOST=0.0.0.0" >> .env
echo "PORT=5000" >> .env
echo "DEBUG=false" >> .env

# Set up firewall rules
echo "🔥 Configuring firewall..."
sudo ufw allow 5000/tcp  # Flask API
sudo ufw allow 11434/tcp # Ollama API (optional, for debugging)
sudo ufw allow ssh
sudo ufw --force enable

# Create systemd service for auto-start
echo "🚀 Creating systemd service..."
sudo tee /etc/systemd/system/invoice-ai-privacy.service > /dev/null <<EOF
[Unit]
Description=Invoice AI Privacy Service
Requires=docker.service
After=docker.service

[Service]
Type=forking
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/invoice-ai-privacy
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
sudo systemctl daemon-reload
sudo systemctl enable invoice-ai-privacy.service

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Logout and login again (for Docker group changes)"
echo "2. Start the service: sudo systemctl start invoice-ai-privacy"
echo "3. Check status: sudo systemctl status invoice-ai-privacy"
echo "4. View logs: docker-compose logs -f"
echo ""
echo "📡 Service will be available at:"
echo "   - API: http://$(curl -s ifconfig.me):5000"
echo "   - Health: http://$(curl -s ifconfig.me):5000/health"
echo ""
echo "🔧 Useful commands:"
echo "   - Start: sudo systemctl start invoice-ai-privacy"
echo "   - Stop: sudo systemctl stop invoice-ai-privacy"
echo "   - Restart: sudo systemctl restart invoice-ai-privacy"
echo "   - Logs: cd ~/invoice-ai-privacy && docker-compose logs -f"
echo ""

# Final system information
echo "💾 System resources:"
free -h
echo ""
df -h
echo ""
echo "🎉 Oracle Cloud setup complete! The service will auto-start on boot."