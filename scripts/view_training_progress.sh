#!/bin/bash
# View Training Progress with Epoch and Accuracy Information

echo ""
echo "================================================================================"
echo "  DISTILBERT TRAINING PROGRESS - DETAILED VIEW"
echo "================================================================================"
echo ""

# Check if training process is running
if ps aux | grep -q "[t]rain_distilbert.py"; then
    echo "✅ Training Status: RUNNING"
    echo ""
    ps aux | grep "[t]rain_distilbert.py" | awk '{print "   PID: " $2 "\n   CPU: " $3 "% \n   Memory: " $4 "% \n   Running Time: " $10}'
    echo ""
else
    echo "❌ Training Status: NOT RUNNING"
    echo ""
fi

echo "================================================================================"
echo "  TRAINING RESULTS SUMMARY"
echo "================================================================================"
echo ""

# Check for training results file (created when training completes)
if [ -f "../models/distilbert-sentiment/training_results.json" ]; then
    echo "🎉 TRAINING COMPLETED!"
    echo ""
    cat ../models/distilbert-sentiment/training_results.json | python3 -m json.tool
else
    echo "Training in progress... Showing epoch summaries from output:"
    echo ""

    # Try to get epoch information from process if available
    # This shows the key metrics from training
    echo "To see live training output with epochs and accuracies, use:"
    echo ""
    echo "   python3 view_training_output.py"
    echo ""
    echo "Or check specific metrics:"
    echo "   - EPOCH information"
    echo "   - Training Accuracy"
    echo "   - Validation Accuracy"
    echo "   - Best model checkpoints"
fi

echo ""
echo "================================================================================"
echo "  MODEL CHECKPOINTS"
echo "================================================================================"
echo ""

# Check for saved model
if [ -d "../models/distilbert-sentiment/best_model" ]; then
    echo "✅ Best Model Saved"
    MODEL_TIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" ../models/distilbert-sentiment/best_model 2>/dev/null || stat -c "%y" ../models/distilbert-sentiment/best_model 2>/dev/null | cut -d'.' -f1)
    echo "   Last Updated: $MODEL_TIME"
else
    echo "⏳ Best model not yet saved"
fi

echo ""

# Show any checkpoints
if ls ../models/distilbert-sentiment/checkpoint-* 1> /dev/null 2>&1; then
    CHECKPOINT_COUNT=$(ls -d ../models/distilbert-sentiment/checkpoint-* 2>/dev/null | wc -l)
    echo "📁 Checkpoints: $CHECKPOINT_COUNT found"
    echo ""
    echo "   Recent checkpoints:"
    ls -lt ../models/distilbert-sentiment/checkpoint-* | head -3 | awk '{print "   - " $NF}' | xargs -I {} basename {}
else
    echo "⏳ No checkpoints saved yet"
fi

echo ""
echo "================================================================================"
echo ""
