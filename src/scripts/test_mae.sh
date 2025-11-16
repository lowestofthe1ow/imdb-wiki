echo "---------- IMDB-WIKI + APPA-REAL ----------"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/IMDBWIKI_CHALEARN_epoch=11_val_loss=2.94.ckpt" \
