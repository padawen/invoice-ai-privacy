# Invoice AI Privacy - Deployment Guide

## Oracle Cloud Free Tier Deployment

### Prerequisites
- Oracle Cloud Free Tier account
- VM.Standard.A1.Flex instance (ARM64, 4 OCPU, 24GB RAM)
- Ubuntu 22.04 LTS

### Quick Setup

1. **Create Oracle Cloud VM**:
   - Instance type: VM.Standard.A1.Flex
   - Image: Ubuntu 22.04 LTS (ARM64)
   - Configuration: 4 OCPU, 24GB RAM
   - Boot volume: 200GB
   - Networking: Allow HTTP/HTTPS traffic

2. **Run setup script**:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/your-username/invoice-ai-privacy/main/deployment/oracle_setup.sh | bash
   ```

3. **Start the service**:
   ```bash
   # Logout and login again for Docker group changes
   exit
   # SSH back in
   sudo systemctl start invoice-ai-privacy
   ```

### Manual Deployment

1. **Connect to your Oracle Cloud VM**:
   ```bash
   ssh -i ~/.ssh/id_rsa ubuntu@your-vm-ip
   ```

2. **Clone and build**:
   ```bash
   git clone https://github.com/your-username/invoice-ai-privacy.git
   cd invoice-ai-privacy
   cp .env.example .env
   docker-compose up --build -d
   ```

### Configuration

**Environment Variables** (`.env`):
```bash
# Production settings for Oracle Cloud
OLLAMA_MODEL=phi3:mini  # or llama2:7b for better accuracy
HOST=0.0.0.0
PORT=5000
DEBUG=false
OCR_LANGUAGE=eng
MAX_FILE_SIZE=52428800
```

**Firewall Rules**:
```bash
sudo ufw allow 5000/tcp   # API access
sudo ufw allow ssh        # SSH access
sudo ufw enable
```

### Monitoring

**Check service status**:
```bash
sudo systemctl status invoice-ai-privacy
```

**View logs**:
```bash
cd ~/invoice-ai-privacy
docker-compose logs -f
```

**Health check**:
```bash
curl http://localhost:5000/health
```

**Resource monitoring**:
```bash
# Memory usage
free -h

# Disk usage
df -h

# Docker stats
docker stats
```

### Performance Optimization

**For 24GB RAM Oracle Cloud instance**:

1. **Use Phi3 Mini model** (2GB RAM):
   ```bash
   echo "OLLAMA_MODEL=phi3:mini" >> .env
   ```

2. **Monitor memory usage**:
   ```bash
   # Check available memory
   free -h

   # If running low, restart the service
   sudo systemctl restart invoice-ai-privacy
   ```

3. **Optimize for ARM64**:
   ```bash
   # The Docker image is optimized for ARM64
   # No additional configuration needed
   ```

### Troubleshooting

**Common Issues**:

1. **Out of memory**:
   - Switch to smaller model: `phi3:mini`
   - Restart services: `sudo systemctl restart invoice-ai-privacy`

2. **Model download fails**:
   ```bash
   # Manual model download
   docker-compose exec invoice-ai-privacy ollama pull phi3:mini
   ```

3. **Service won't start**:
   ```bash
   # Check Docker status
   sudo systemctl status docker

   # Check logs
   docker-compose logs

   # Restart everything
   sudo systemctl restart docker
   sudo systemctl restart invoice-ai-privacy
   ```

4. **API not accessible**:
   - Check firewall: `sudo ufw status`
   - Check service: `curl http://localhost:5000/health`
   - Check public IP in Oracle Cloud console

### Scaling

**For higher workloads**:

1. **Upgrade to larger model**:
   ```bash
   echo "OLLAMA_MODEL=llama2:7b" >> .env
   sudo systemctl restart invoice-ai-privacy
   ```

2. **Add more compute (paid tier)**:
   - Scale up to more OCPUs
   - Add GPU instances for faster processing

### Security

**Production security checklist**:

1. **Add API authentication**:
   ```bash
   echo "API_KEY=your-secure-random-key" >> .env
   ```

2. **Use HTTPS** (with reverse proxy):
   ```bash
   # Install nginx
   sudo apt install nginx

   # Configure SSL with Let's Encrypt
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

3. **Restrict access**:
   ```bash
   # Allow only specific IPs
   sudo ufw allow from YOUR_IP to any port 5000
   ```

### Cost Monitoring

**Oracle Cloud Free Tier limits**:
- ✅ Compute: Always free (ARM64)
- ✅ Storage: 200GB boot volume (free)
- ✅ Network: 10TB outbound/month (free)
- ⚠️ Monitor usage in Oracle Cloud Console

### Backup

**Backup your setup**:
```bash
# Backup configuration and models
tar -czf invoice-ai-backup.tar.gz \
  ~/invoice-ai-privacy/.env \
  ~/.ollama/
```

**Restore**:
```bash
# Restore configuration
tar -xzf invoice-ai-backup.tar.gz -C ~/
```