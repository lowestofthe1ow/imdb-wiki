# --dataset can be IMDBWIKI, APPAREAL, or CHALEARN

uv run \
    -m src.model.train.train_dex \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 20 \
    --max-epochs 100 \
    --method "MVL" \
    --dataset "CHALEARN" \
    --dataset-path "data/chalearn_2015/" \
    --outfile-prefix "MVL_IMDBWIKI" \
    # --checkpoint-path "checkpoints/DEX_IMDBWIKI_epoch=26_val_loss=3.36.ckpt" \
    # --use-weighted-ce \

