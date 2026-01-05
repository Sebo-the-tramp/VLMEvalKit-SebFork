RUN_NAME="run_10_general"
QUANTITY="10K"

if [ -d "/data0/sebastian.cavada/datasets/simulations_v4" ]; then
    INPUT_PATH="/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output"
    OUTPUT_PATH="/home/cavadalab/Documents/scsv/papers/computational_physiscs/inria_version_3/VLMEvalKit-SebFork/vqa"
    IMAGE_ROOT="/data0/sebastian.cavada/datasets/simulations_v4/dl3dv"
    FOLDER="/"
    FILE_NAME="NewtPhys_MultiImage.tsv"
else
    source "/home/it4i-thvu/seb_dev/.telegram_bot.env"
    INPUT_PATH="/mnt/proj1/eu-25-92/tiny_vqa_creation/output"
    OUTPUT_PATH="/mnt/proj1/eu-25-92/tiny_vqa_creation/output"
    IMAGE_ROOT="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv"
    FOLDER="/${RUN_NAME}/"
    FILE_NAME="test_${RUN_NAME}_${QUANTITY}.tsv"
fi

python convert_tsv.py ${INPUT_PATH}/${RUN_NAME}/test_${RUN_NAME}_${QUANTITY}.json ${OUTPUT_PATH}${FOLDER}${FILE_NAME} \
    --image-root ${IMAGE_ROOT}
