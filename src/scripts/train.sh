# --dataset can be IMDBWIKI, APPAREAL, or CHALEARN

uv run \
    -m src.model.train.train_dex \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method "DEX" \
    --dataset "CHALEARN" \
    --dataset-path "data/chalearn_2015/" \
    --outfile-prefix "DEX_IMDBWIKI" \
    --checkpoint-path "checkpoints/DEX_IMDBWIKI_epoch=06_val_loss=3.86.ckpt" \
    # --use-weighted-ce \

