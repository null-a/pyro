OUTPUT_FILE="`dirname $0`/tr2_obj11_s.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/p0few88684ojqh1/tr2_obj11_s.npz"
fi
