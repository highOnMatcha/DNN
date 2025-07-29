#!/bin/bash
# Comprehensive training script for large LSTM models on all stock datasets
# This script trains large models with advanced configurations on all available stocks

echo "🚀 Starting comprehensive LSTM training on large models and datasets"
echo "=================================================="

# Configuration
MODELS=("lstm-large" "lstm-attention")
SYMBOLS=("AAPL" "GOOGL" "META" "TSLA" "NVDA" "MSFT" "AMZN" "NFLX" "AMD" "INTC")
CONFIG="production"
DATA_CONFIG="large"

# Training parameters
BATCH_SIZE=128
EPOCHS=200
LEARNING_RATE=0.0005

echo "📊 Training Configuration:"
echo "  Models: ${MODELS[*]}"
echo "  Symbols: ${SYMBOLS[*]}"
echo "  Training Config: $CONFIG"
echo "  Data Config: $DATA_CONFIG"
echo "  Expected training time: ~4-6 hours"
echo "=================================================="

# Create results directory
mkdir -p training_results
RESULTS_FILE="training_results/large_model_training_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Results will be logged to: $RESULTS_FILE"
echo "Starting training at: $(date)" | tee $RESULTS_FILE

# Train each model on each symbol
total_experiments=$((${#MODELS[@]} * ${#SYMBOLS[@]}))
current_experiment=0

for model in "${MODELS[@]}"; do
    echo "" | tee -a $RESULTS_FILE
    echo "🧠 Training Model: $model" | tee -a $RESULTS_FILE
    echo "================================" | tee -a $RESULTS_FILE
    
    for symbol in "${SYMBOLS[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo "" | tee -a $RESULTS_FILE
        echo "📈 [$current_experiment/$total_experiments] Training $model on $symbol" | tee -a $RESULTS_FILE
        echo "Started at: $(date)" | tee -a $RESULTS_FILE
        
        # Run training with advanced configuration
        python3 src/train.py \
            --model "$model" \
            --symbol "$symbol" \
            --config "$CONFIG" \
            --data-config "$DATA_CONFIG" \
            --wandb 2>&1 | tee -a $RESULTS_FILE
        
        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed $model on $symbol" | tee -a $RESULTS_FILE
        else
            echo "❌ Failed training $model on $symbol" | tee -a $RESULTS_FILE
        fi
        
        echo "Completed at: $(date)" | tee -a $RESULTS_FILE
        echo "----------------------------------------" | tee -a $RESULTS_FILE
    done
done

echo "" | tee -a $RESULTS_FILE
echo "🎉 All training experiments completed!" | tee -a $RESULTS_FILE
echo "Finished at: $(date)" | tee -a $RESULTS_FILE
echo "Total experiments: $total_experiments" | tee -a $RESULTS_FILE
echo "Results logged to: $RESULTS_FILE" | tee -a $RESULTS_FILE

# Generate summary
echo "" | tee -a $RESULTS_FILE
echo "📊 Training Summary:" | tee -a $RESULTS_FILE
echo "===================" | tee -a $RESULTS_FILE

# Count successful completions
successful_count=$(grep -c "✅ Successfully completed" $RESULTS_FILE)
failed_count=$(grep -c "❌ Failed training" $RESULTS_FILE)

echo "Successful trainings: $successful_count" | tee -a $RESULTS_FILE
echo "Failed trainings: $failed_count" | tee -a $RESULTS_FILE

# Show model files created
echo "" | tee -a $RESULTS_FILE
echo "📁 Generated model files:" | tee -a $RESULTS_FILE
find models/ -name "*.pt" -newer $RESULTS_FILE 2>/dev/null | tee -a $RESULTS_FILE

echo ""
echo "🔗 View your experiments on WandB: https://wandb.ai/khalil-epfl/stock-prediction"
echo "🚀 Training pipeline completed successfully!"
