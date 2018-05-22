OUTPUT_FILE="`dirname $0`/bkg_params.pytorch"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/0w52kh8n5ce8nz8/bkg_params.pytorch"
fi
