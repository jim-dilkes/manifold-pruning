git clone https://github.com/soskek/bookcorpus.git data/bookcorpus
pip install -r data/bookcorpus/requirements.txt
python data/bookcorpus/download_files.py --list data/bookcorpus/url_list.jsonl --out data/bookcorpus/out_txts --trash-bad-count
