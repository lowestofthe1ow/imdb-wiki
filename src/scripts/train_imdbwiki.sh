uv run \
    -m src.model.train.imdbwiki \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method "DEX" \
    --dataset "APPAREAL" \
    --dataset-path "data/appa-real-release/" \
    --outfile-prefix "IMDBWIKI_APPAREAL" \
    --checkpoint-path "checkpoints/IMDBWIKI_DEX_epoch=28_val_loss=3.36.ckpt" \
    # --use-weighted-ce
    # --batch-size 64 \
    # --csv-path "/kaggle/input/imdb-wiki-cleaned/imdbwiki.csv" \
    # --data-path "/kaggle/input/imdb-wiki-cleaned"
