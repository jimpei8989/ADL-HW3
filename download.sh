REMOTE_URLs=("wjpei.csie.org:29333/" "linux9.csie.ntu.edu.tw:29333/" "linux10.csie.ntu.edu.tw:29333/" "linux11.csie.ntu.edu.tw:29333/" "linux12.csie.ntu.edu.tw:29333/")
REMOTE_FILE_PATH="model.zip"
LOCAL_MODEL_DIR="models/"
DOWNLOAD_PATH=${LOCAL_MODEL_DIR}"model.zip"

function download() {
    wget $1$REMOTE_FILE_PATH -O ${DOWNLOAD_PATH}

    if [[ $? -ne 0 ]]; then
        echo "Fail to download from ${1}......"
        return -1
    else
        echo "Successfully download model.zip from ${1}"
        (cd ${LOCAL_MODEL_DIR} && (yes | unzip model.zip))
        return 0
    fi
}

mkdir -p $LOCAL_MODEL_DIR
for url in ${REMOTE_URLs[@]}; do
    download ${url}
    if [[ $? -eq 0 ]]; then
        break
    fi
done

ls -al $LOCAL_MODEL_DIR
