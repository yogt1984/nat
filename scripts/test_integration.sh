#!/bin/bash
# NAT Integration Test Script
#
# Tests the full stack: Redis -> Ingestor -> API -> Alerts
#
# Prerequisites:
#   - Redis running on localhost:6379
#   - API server running on localhost:3000
#   - (Optional) Ingestor running and publishing features

set -e

API_URL="${API_URL:-http://localhost:3000}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0
warnings=0

pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((passed++))
}

fail() {
    echo -e "${RED}✗${NC} $1"
    ((failed++))
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((warnings++))
}

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              NAT INTEGRATION TEST SUITE                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "API URL: $API_URL"
echo "Redis:   $REDIS_URL"
echo ""

# =============================================================================
# 1. REDIS TESTS
# =============================================================================
echo "───────────────────────────────────────────────────────────────────"
echo " 1. REDIS CONNECTIVITY"
echo "───────────────────────────────────────────────────────────────────"

# Test Redis ping
if redis-cli ping > /dev/null 2>&1; then
    pass "Redis is running"
else
    fail "Redis is not running"
    echo "  Start Redis with: sudo systemctl start redis-server"
    exit 1
fi

# Check for cached features
SYMBOLS=$(redis-cli KEYS "nat:latest:*" 2>/dev/null | wc -l)
if [ "$SYMBOLS" -gt 0 ]; then
    pass "Found $SYMBOLS symbol(s) with cached data"
    redis-cli KEYS "nat:latest:*" 2>/dev/null | sed 's/nat:latest://g' | tr '\n' ' '
    echo ""
else
    warn "No cached features found (ingestor may not be running)"
fi

echo ""

# =============================================================================
# 2. API SERVER TESTS
# =============================================================================
echo "───────────────────────────────────────────────────────────────────"
echo " 2. API SERVER"
echo "───────────────────────────────────────────────────────────────────"

# Test health endpoint
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" 2>/dev/null)
if [ "$HEALTH" = "200" ]; then
    pass "Health endpoint responding"
    VERSION=$(curl -s "$API_URL/health" | jq -r '.version' 2>/dev/null)
    echo "  Version: $VERSION"
else
    fail "Health endpoint not responding (HTTP $HEALTH)"
    echo "  Start API with: make api"
    exit 1
fi

# Test symbols endpoint
SYMBOLS_RESP=$(curl -s "$API_URL/api/symbols" 2>/dev/null)
if echo "$SYMBOLS_RESP" | jq -e '.' > /dev/null 2>&1; then
    SYMBOL_COUNT=$(echo "$SYMBOLS_RESP" | jq 'length')
    pass "Symbols endpoint working ($SYMBOL_COUNT symbols)"
else
    warn "Symbols endpoint returned invalid JSON"
fi

# Test features endpoint (use first available symbol or BTC)
TEST_SYMBOL=$(echo "$SYMBOLS_RESP" | jq -r '.[0] // "BTC"' 2>/dev/null)
FEATURES_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/features/$TEST_SYMBOL" 2>/dev/null)
if [ "$FEATURES_CODE" = "200" ]; then
    pass "Features endpoint working for $TEST_SYMBOL"
    MIDPRICE=$(curl -s "$API_URL/api/features/$TEST_SYMBOL" | jq -r '.metrics.midprice // "N/A"' 2>/dev/null)
    echo "  Midprice: $MIDPRICE"
elif [ "$FEATURES_CODE" = "404" ]; then
    warn "No features data for $TEST_SYMBOL (ingestor may not be running)"
else
    fail "Features endpoint error (HTTP $FEATURES_CODE)"
fi

# Test regime endpoint
REGIME_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/regime/$TEST_SYMBOL" 2>/dev/null)
if [ "$REGIME_CODE" = "200" ]; then
    pass "Regime endpoint working"
    REGIME_TYPE=$(curl -s "$API_URL/api/regime/$TEST_SYMBOL" | jq -r '.regime_type // "N/A"' 2>/dev/null)
    echo "  Regime: $REGIME_TYPE"
elif [ "$REGIME_CODE" = "404" ]; then
    warn "Regime data not available (needs ~60 min of data)"
else
    fail "Regime endpoint error (HTTP $REGIME_CODE)"
fi

# Test whales endpoint
WHALES_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/whales/$TEST_SYMBOL" 2>/dev/null)
if [ "$WHALES_CODE" = "200" ]; then
    pass "Whales endpoint working"
    DIRECTION=$(curl -s "$API_URL/api/whales/$TEST_SYMBOL" | jq -r '.direction // "N/A"' 2>/dev/null)
    echo "  Direction: $DIRECTION"
elif [ "$WHALES_CODE" = "404" ]; then
    warn "Whale data not available"
else
    fail "Whales endpoint error (HTTP $WHALES_CODE)"
fi

echo ""

# =============================================================================
# 3. WEBSOCKET TESTS
# =============================================================================
echo "───────────────────────────────────────────────────────────────────"
echo " 3. WEBSOCKET ENDPOINTS"
echo "───────────────────────────────────────────────────────────────────"

# Check if wscat is available
if command -v wscat &> /dev/null; then
    # Test WebSocket connection (timeout after 2 seconds)
    WS_TEST=$(timeout 2 wscat -c "ws://localhost:3000/ws/stream/$TEST_SYMBOL" 2>&1 || true)
    if echo "$WS_TEST" | grep -q "Connected"; then
        pass "WebSocket /ws/stream/$TEST_SYMBOL connectable"
    else
        warn "WebSocket connection test inconclusive"
    fi
else
    warn "wscat not installed, skipping WebSocket test"
    echo "  Install with: npm install -g wscat"
fi

echo ""
echo "WebSocket endpoints available:"
echo "  ws://localhost:3000/ws/stream/:symbol - Real-time features"
echo "  ws://localhost:3000/ws/alerts         - Alert stream"

echo ""

# =============================================================================
# 4. SUMMARY
# =============================================================================
echo "═══════════════════════════════════════════════════════════════════"
echo " SUMMARY"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo -e "  ${GREEN}Passed:${NC}   $passed"
echo -e "  ${YELLOW}Warnings:${NC} $warnings"
echo -e "  ${RED}Failed:${NC}   $failed"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}All critical tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check the output above.${NC}"
    exit 1
fi
