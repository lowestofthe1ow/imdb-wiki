# --dataset can be IMDBWIKI, APPAREAL, or CHALEARN

uv run \
    -m src.model.train.train_dex \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method "DEX" \
    --dataset "IMDBWIKI" \
    --dataset-path "data/imdb-wiki/" \
    --outfile-prefix "DEX_IMDBWIKI" \

    # --checkpoint-path "checkpoints/IMDBWIKI_DEX_epoch=28_val_loss=3.36.ckpt" \
    # --use-weighted-ce \

