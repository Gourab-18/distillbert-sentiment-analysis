#!/bin/bash
# Quick Training Status Check Script

echo ""
echo "================================"
echo "  QUICK TRAINING STATUS CHECK"
echo "================================"
echo ""

# Check if training process is running
if ps aux | grep -q "[t]rain_distilbert.py"; then
    echo "✅ Training is RUNNING"
    echo ""
    ps aux | grep "[t]rain_distilbert.py" | awk '{print "   PID: " $2 "\n   CPU: " $3 "% \n   Memory: " $4 "% \n   Time: " $10}'
else
    echo "❌ Training is NOT RUNNING"
fi

echo ""
echo "--------------------------------"

# Check model directory
if [ -d "../models/distilbert-sentiment/best_model" ]; then
    echo "✅ Best model saved"
else
    echo "⏳ Best model not yet saved"
fi

# Check results file
if [ -f "../models/distilbert-sentiment/training_results.json" ]; then
    echo "✅ Training completed!"
    echo ""
    echo "Results:"
    cat ../models/distilbert-sentiment/training_results.json | python3 -m json.tool 2>/dev/null || echo "   (Could not parse results file)"
else
    echo "⏳ Training in progress..."
fi

echo ""
echo "================================"
echo ""
