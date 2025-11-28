echo -e "\nOriginal Caffe model (IMDB-WIKI), testing on ChaLearn test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/Caffe_IMDBWIKI.pth" \

echo -e "\nOriginal Caffe model (IMDB-WIKI + ChaLearn), testing on ChaLearn test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/Caffe_IMDBWIKI_CHALEARN.pth" \

echo -e "\nOriginal Caffe model (IMDB-WIKI), testing on APPA-REAL test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "APPAREAL" \
    --model-path "checkpoints/Caffe_IMDBWIKI.pth" \

echo -e "\nOriginal Caffe model (IMDB-WIKI + ChaLearn), testing on APPA-REAL test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "APPAREAL" \
    --model-path "checkpoints/Caffe_IMDBWIKI_CHALEARN.pth" \

echo -e "\nIMDB-WIKI, testing on ChaLearn test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/DEX_IMDBWIKI_epoch=26_val_loss=3.36.ckpt" \

echo -e "\nIMDB-WIKI, testing on APPA-REAL test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "APPAREAL" \
    --model-path "checkpoints/DEX_IMDBWIKI_epoch=26_val_loss=3.36.ckpt" \

echo -e "\nIMDB-WIKI + ChaLearn train+val set, testing on ChaLearn test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/IMDBWIKI_CHALEARN_epoch=13_val_loss=2.81.ckpt" \

echo -e "\nIMDB-WIKI + APPA-REAL train+val set, testing on ChaLearn test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/IMDBWIKI_APPAREAL_epoch=09_val_loss=2.96.ckpt" \

echo -e "\nIMDB-WIKI + ChaLearn train+val set, testing on APPA-REAL test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "APPAREAL" \
    --model-path "checkpoints/IMDBWIKI_CHALEARN_epoch=13_val_loss=2.81.ckpt" \

echo -e "\nIMDB-WIKI + APPA-REAL train+val set, testing on APPA-REAL test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "APPAREAL" \
    --model-path "checkpoints/IMDBWIKI_APPAREAL_epoch=09_val_loss=2.96.ckpt" \

# This is our best recreation of DEX so far, with MAE of about 3.57 on ChaLearn LAP '15
# https://drive.google.com/drive/folders/1dFLn0RZ2FlvU8WjqwJ-YrOMor_SwTPfP?usp=sharing

# TODO: Execute this
# (I think my `data/` directory structure is different from yours -- Justin to Lorenz)

echo -e "\nIMDB-WIKI, testing on ChaLearn test set"

uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/AMRL_IMDBWIKI_epoch=07_val_loss=10.19.ckpt" \
