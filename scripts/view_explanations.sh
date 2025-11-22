#!/bin/bash

# View SHAP Explanations Helper Script

EXPLANATIONS_DIR="../reports/explanations"

echo "=========================================="
echo "SHAP Explanations Viewer"
echo "=========================================="
echo ""

# Check if directory exists
if [ ! -d "$EXPLANATIONS_DIR" ]; then
    echo "❌ Explanations directory not found!"
    echo "Please run: python explain.py first"
    exit 1
fi

echo "Available options:"
echo ""
echo "1. Open explanations folder in Finder"
echo "2. Open first 5 force plots (HTML) in browser"
echo "3. Open first 5 waterfall plots (PNG)"
echo "4. Open first 5 text plots (PNG)"
echo "5. Open all summary plots"
echo "6. Open summary report (Markdown)"
echo "7. List all files"
echo "8. Open specific example by number (0-29)"
echo ""

read -p "Enter your choice (1-8): " choice

case $choice in
    1)
        echo "📂 Opening explanations folder..."
        open "$EXPLANATIONS_DIR"
        ;;
    2)
        echo "🌐 Opening first 5 force plots in browser..."
        for file in $(ls "$EXPLANATIONS_DIR/force_plots/" | head -5); do
            open "$EXPLANATIONS_DIR/force_plots/$file"
            sleep 0.5
        done
        ;;
    3)
        echo "📊 Opening first 5 waterfall plots..."
        for file in $(ls "$EXPLANATIONS_DIR/waterfall_plots/" | head -5); do
            open "$EXPLANATIONS_DIR/waterfall_plots/$file"
            sleep 0.5
        done
        ;;
    4)
        echo "📝 Opening first 5 text plots..."
        for file in $(ls "$EXPLANATIONS_DIR/text_plots/" | head -5); do
            open "$EXPLANATIONS_DIR/text_plots/$file"
            sleep 0.5
        done
        ;;
    5)
        echo "📈 Opening summary plots..."
        open "$EXPLANATIONS_DIR/category_distribution.png"
        open "$EXPLANATIONS_DIR/confidence_distribution.png"
        open "$EXPLANATIONS_DIR/text_length_distribution.png"
        ;;
    6)
        echo "📄 Opening summary report..."
        open "$EXPLANATIONS_DIR/explanations_summary.md"
        ;;
    7)
        echo ""
        echo "📁 Force Plots (HTML):"
        ls "$EXPLANATIONS_DIR/force_plots/"
        echo ""
        echo "📊 Waterfall Plots (PNG):"
        ls "$EXPLANATIONS_DIR/waterfall_plots/"
        echo ""
        echo "📝 Text Plots (PNG):"
        ls "$EXPLANATIONS_DIR/text_plots/"
        echo ""
        echo "📈 Summary Files:"
        ls "$EXPLANATIONS_DIR"/*.png "$EXPLANATIONS_DIR"/*.md "$EXPLANATIONS_DIR"/*.json 2>/dev/null
        ;;
    8)
        read -p "Enter example number (0-29): " num

        # Find files with example number pattern
        force_file=$(ls "$EXPLANATIONS_DIR/force_plots/" | grep "example_0*${num}_")
        waterfall_file=$(ls "$EXPLANATIONS_DIR/waterfall_plots/" | grep "example_0*${num}_")
        text_file=$(ls "$EXPLANATIONS_DIR/text_plots/" | grep "example_0*${num}_")

        if [ -n "$force_file" ]; then
            echo ""
            echo "Opening visualizations for example $num..."
            open "$EXPLANATIONS_DIR/force_plots/$force_file"
            open "$EXPLANATIONS_DIR/waterfall_plots/$waterfall_file"
            open "$EXPLANATIONS_DIR/text_plots/$text_file"
        else
            echo "❌ Example $num not found!"
        fi
        ;;
    *)
        echo "❌ Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"
