REMOTE_URLs=("wjpei.csie.org:29333/" "linux9.csie.ntu.edu.tw:29333" "linux10.csie.ntu.edu.tw:29333" "linux11.csie.ntu.edu.tw:29333" "linux12.csie.ntu.edu.tw:29333")
REMOTE_FILE_PATH="/model.zip"
LOCAL_MODEL_DIR="models/model/"

function download() {
    wget $1$REMOTE_FILE_PATH -O ${LOCAL_MODEL_DIR}

    if [[ $? -ne 0 ]]; then
        echo "Fail to download from ${1}......"
        return -1
    else
        echo "Successfully download model.zip from ${1}"
        cd ${LOCAL_MODEL_DIR} && unzip model.zip
        return 0
    fi
}

for url in ${REMOTE_URLs[@]}; do
    download ${url}
    if [[ $? -e 0 ]]; then
        break
    fi
done
