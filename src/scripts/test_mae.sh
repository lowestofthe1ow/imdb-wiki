echo "IMDB-WIKI + ChaLearn train+val set, testing on ChaLearn test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/IMDBWIKI_CHALEARN_epoch=13_val_loss=2.81.ckpt" \

echo "IMDB-WIKI + APPA-REAL train+val set, testing on ChaLearn test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/IMDBWIKI_APPAREAL_epoch=09_val_loss=2.96.ckpt" \

echo "IMDB-WIKI + ChaLearn train+val set, testing on APPA-REAL test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "APPAREAL" \
    --model-path "checkpoints/IMDBWIKI_CHALEARN_epoch=13_val_loss=2.81.ckpt" \

echo "IMDB-WIKI + APPA-REAL train+val set, testing on APPA-REAL test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "APPAREAL" \
    --model-path "checkpoints/IMDBWIKI_APPAREAL_epoch=09_val_loss=2.96.ckpt" \

# This is our best recreation of DEX so far, with MAE of about 3.57 on ChaLearn LAP '15
# https://drive.google.com/drive/folders/1dFLn0RZ2FlvU8WjqwJ-YrOMor_SwTPfP?usp=sharing
