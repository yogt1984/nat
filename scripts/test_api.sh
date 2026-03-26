#!/bin/bash
# Test NAT API endpoints

API_URL="${API_URL:-http://localhost:3000}"

echo "Testing NAT API at $API_URL"
echo "================================"
echo ""

# Health check
echo "1. Health check:"
curl -s "$API_URL/health" | jq .
echo ""

# Active symbols
echo "2. Active symbols:"
curl -s "$API_URL/api/symbols" | jq .
echo ""

# Features endpoint (BTC)
echo "3. Features (BTC):"
curl -s "$API_URL/api/features/BTC" | jq .
echo ""

# Regime endpoint
echo "4. Regime (BTC):"
curl -s "$API_URL/api/regime/BTC" | jq .
echo ""

# Whales endpoint
echo "5. Whales (BTC):"
curl -s "$API_URL/api/whales/BTC" | jq .
echo ""

echo "================================"
echo "WebSocket test commands:"
echo "  wscat -c ws://localhost:3000/ws/stream/BTC"
echo "  wscat -c ws://localhost:3000/ws/alerts"
echo ""
echo "Redis test commands:"
echo "  redis-cli SUBSCRIBE nat:features:BTC"
echo "  redis-cli SUBSCRIBE nat:alerts"
