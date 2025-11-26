CSV="checkpoints/checkpoint_captions.csv"

tail -n +2 "$CSV" | while IFS=, read -ra line; do
    MODEL_PATH="${line[0]}"

    if [ -n "$MODEL_PATH" ]; then
        echo "Testing $MODEL_PATH on APPAREAL"
        uv run \
            -m src.model.test.mean_absolute_error \
            --testing-set "APPAREAL" \
            --model-path "$MODEL_PATH" \
            --model-captions-path "$CSV"
        echo "----------"

        echo "Testing $MODEL_PATH on CHALEARN"
        uv run \
            -m src.model.test.mean_absolute_error \
            --testing-set "CHALEARN" \
            --model-path "$MODEL_PATH" \
            --model-captions-path "$CSV"
        echo "----------"
    fi
done
