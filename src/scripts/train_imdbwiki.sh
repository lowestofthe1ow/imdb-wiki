uv run \
    -m src.model.train.imdbwiki \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method DEX \
    --csv-path "data/imdbwiki.csv" \
    --data-path "data/imdbwiki/"
    # --batch-size 64 \
    # --csv-path "/kaggle/input/imdb-wiki-cleaned/imdbwiki.csv" \
    # --data-path "/kaggle/input/imdb-wiki-cleaned"