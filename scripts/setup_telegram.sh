#!/bin/bash
# Setup Telegram Bot for NAT Alerts
#
# This script helps you configure Telegram alerts.

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              NAT TELEGRAM ALERT SETUP                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check for existing .env file
ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    echo "Found existing .env file"
    source "$ENV_FILE"
fi

# Check if already configured
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    echo "Telegram is already configured:"
    echo "  Bot Token: ${TELEGRAM_BOT_TOKEN:0:10}..."
    echo "  Chat ID: $TELEGRAM_CHAT_ID"
    echo ""
    read -p "Do you want to reconfigure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing configuration."
        exit 0
    fi
fi

echo ""
echo "STEP 1: Create a Telegram Bot"
echo "─────────────────────────────"
echo "1. Open Telegram and search for @BotFather"
echo "2. Send /newbot"
echo "3. Follow the prompts to create your bot"
echo "4. Copy the bot token (looks like: 123456789:ABCdef...)"
echo ""

read -p "Enter your bot token: " BOT_TOKEN

if [ -z "$BOT_TOKEN" ]; then
    echo "Error: Bot token is required"
    exit 1
fi

echo ""
echo "STEP 2: Get your Chat ID"
echo "────────────────────────"
echo "1. Start a chat with your new bot"
echo "2. Send any message to the bot"
echo "3. Open this URL in your browser:"
echo "   https://api.telegram.org/bot${BOT_TOKEN}/getUpdates"
echo "4. Find 'chat':{'id':XXXXXXXX} in the response"
echo ""

read -p "Enter your chat ID: " CHAT_ID

if [ -z "$CHAT_ID" ]; then
    echo "Error: Chat ID is required"
    exit 1
fi

# Test the configuration
echo ""
echo "Testing configuration..."
RESPONSE=$(curl -s "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -d "chat_id=${CHAT_ID}" \
    -d "text=🟢 NAT Alert Service configured successfully!" \
    -d "parse_mode=HTML")

if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "✓ Test message sent successfully!"
else
    echo "✗ Failed to send test message"
    echo "Response: $RESPONSE"
    exit 1
fi

# Save to .env file
echo ""
echo "Saving configuration to .env..."

# Remove old values if they exist
if [ -f "$ENV_FILE" ]; then
    grep -v "TELEGRAM_BOT_TOKEN" "$ENV_FILE" | grep -v "TELEGRAM_CHAT_ID" > "${ENV_FILE}.tmp"
    mv "${ENV_FILE}.tmp" "$ENV_FILE"
fi

echo "TELEGRAM_BOT_TOKEN=${BOT_TOKEN}" >> "$ENV_FILE"
echo "TELEGRAM_CHAT_ID=${CHAT_ID}" >> "$ENV_FILE"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              SETUP COMPLETE!                                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration saved to .env"
echo ""
echo "To start the alert service:"
echo "  source .env && make alerts"
echo ""
echo "Or start the full stack:"
echo "  source .env && make serve_all"
echo ""
