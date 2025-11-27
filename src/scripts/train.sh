# --dataset can be IMDBWIKI, APPAREAL, or CHALEARN
# --method can be DEX, MVL, AMRL
time uv run \
    -m src.model.train.train_dex \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 20 \
    --max-epochs 100 \
    --method "AMRL" \
    --dataset "IMDBWIKI" \
    --data-path "/kaggle/input/imdb-wiki-cleaned" \
    --outfile-prefix "AMRL_IMDBWIKI" \
    # --checkpoint-path "checkpoints/DEX_IMDBWIKI_epoch=26_val_loss=3.36.ckpt" \
    # --use-weighted-ce \
