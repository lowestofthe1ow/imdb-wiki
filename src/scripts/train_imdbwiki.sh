echo Training DEX on Processed IMDB WIKI dataset...
time uv run \
    -m src.model.train.imdbwiki \
    --batch-size 64 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method DEX \
    --csv-path "data/processed-imdb-wiki-dataset/imdbwiki.csv" \
    --data-path "data/processed-imdb-wiki-dataset"

echo Training MVL on Processed IMDB WIKI dataset...
time uv run \
    -m src.model.train.imdbwiki \
    --batch-size 64 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method MVL \
    --csv-path "data/processed-imdb-wiki-dataset/imdbwiki.csv" \
    --data-path "data/processed-imdb-wiki-dataset"

    # --csv-path "/kaggle/input/imdb-wiki-cleaned/imdbwiki.csv" \
    # --data-path "/kaggle/input/imdb-wiki-cleaned"