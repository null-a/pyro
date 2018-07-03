OUTPUT_FILE="`dirname $0`/bkg_tr2_obj11_s.pytorch"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/towzi4z42rzsvnh/params-5000.pytorch"
fi
