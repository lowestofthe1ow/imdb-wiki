python3 make_csv/csv_imdbwiki.py --imdb-path "/kaggle/input/imdb-wiki/imdb_crop" --wiki-path "/kaggle/input/imdb-wiki/wiki_crop"

python3 train_imdbwiki.py --batch-size 64 --workers 4 --min-epochs 0 --max-epochs 200 --method "DEX"
