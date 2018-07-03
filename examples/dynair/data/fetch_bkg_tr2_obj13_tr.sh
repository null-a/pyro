OUTPUT_FILE="`dirname $0`/bkg_tr2_obj13_tr.pytorch"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/ggtcsdcxzf8ijad/params-5000.pytorch"
fi
